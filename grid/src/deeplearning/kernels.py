"""Empirical neural tangent kernels and the scale, shape and alignment of their movement.

The kernel is the Gram matrix of parameter gradients on a fixed probe set. Two
computations follow the torch.func neural tangent kernel tutorial: Jacobian
contraction and NTK-vector products, sections 3.2 and 3.3 of Novak et al.
(2022). For a network with one output, the tutorial's ``trace`` contraction
gives the ``n x n`` kernel.

Centring, the Frobenius product and centred alignment follow Cortes, Mohri and
Rostamizadeh (2012): Equation 1, Lemma 1 and Definition 4. The uncentred
alignment is the one their Section 2.3 attributes to Cristianini et al.

The infinite-width Gram matrix of a bias-free two-layer ReLU network on the
sphere is Equation 4 of Basri et al. (2019).

Every function takes kernels with any leading batch dimensions, so one call
covers all seeds and checkpoints of a cell.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch.func import jacrev, jvp, vjp, vmap

Params = Mapping[str, torch.Tensor]
SingleInput = Callable[[Params, torch.Tensor], torch.Tensor]


CONTRACTIONS = {
    "full": "Naf,Mbf->NMab",
    "trace": "Naf,Maf->NM",
    "diagonal": "Naf,Maf->NMa",
}
BLOCKS = {"full": "NMab->NMab", "trace": "NMKK->NM", "diagonal": "NMKK->NMK"}


def entk_jacobian_contraction(
    fnet_single: SingleInput,
    params: Params,
    x1: torch.Tensor,
    x2: torch.Tensor,
    compute: str = "full",
) -> torch.Tensor:
    """Empirical NTK between ``x1`` and ``x2`` by contracting Jacobians.

    Args:
        fnet_single: Network evaluated on one example, ``(params, x_i) -> logits``.
        params: Parameters, keyed as in ``named_parameters``.
        x1: First batch of inputs, one example per row.
        x2: Second batch of inputs, one example per row.
        compute: ``"full"`` for every logit pair, ``"trace"`` for the sum over
            logits, ``"diagonal"`` for one kernel per logit.

    Returns:
        The kernel, shaped ``(N, M, O, O)``, ``(N, M)`` or ``(N, M, O)``.
    """
    jac1 = [
        j.flatten(2) for j in vmap(jacrev(fnet_single), (None, 0))(params, x1).values()
    ]
    jac2 = [
        j.flatten(2) for j in vmap(jacrev(fnet_single), (None, 0))(params, x2).values()
    ]
    expr = CONTRACTIONS[compute]
    return torch.stack(
        [torch.einsum(expr, j1, j2) for j1, j2 in zip(jac1, jac2, strict=True)]
    ).sum(0)


def entk_ntk_vps(
    fnet_single: SingleInput,
    params: Params,
    x1: torch.Tensor,
    x2: torch.Tensor,
    compute: str = "full",
) -> torch.Tensor:
    """Empirical NTK between ``x1`` and ``x2`` from NTK-vector products.

    Args:
        fnet_single: Network evaluated on one example, ``(params, x_i) -> logits``.
        params: Parameters, keyed as in ``named_parameters``.
        x1: First batch of inputs, one example per row.
        x2: Second batch of inputs, one example per row.
        compute: ``"full"`` for every logit pair, ``"trace"`` for the sum over
            logits, ``"diagonal"`` for one kernel per logit.

    Returns:
        The kernel, shaped ``(N, M, O, O)``, ``(N, M)`` or ``(N, M, O)``.
    """

    def get_ntk(x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        def func_x1(p: Params) -> torch.Tensor:
            return fnet_single(p, x_1)

        def func_x2(p: Params) -> torch.Tensor:
            return fnet_single(p, x_2)

        output, vjp_fn = vjp(func_x1, params)[:2]

        def get_ntk_slice(vec: torch.Tensor) -> torch.Tensor:
            vjps = vjp_fn(vec)
            jvps: torch.Tensor = jvp(func_x2, (params,), vjps)[1]
            return jvps

        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device)
        ntk: torch.Tensor = vmap(get_ntk_slice)(basis)
        return ntk

    full = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    return torch.einsum(BLOCKS[compute], full)


def centre(k: torch.Tensor) -> torch.Tensor:
    """Centre kernel matrices in feature space, Cortes et al. (2012), Equation 1.

    Args:
        k: Kernels of shape ``(..., n, n)``.

    Returns:
        ``H k H`` with ``H = I - 11^T / n``, formed from row, column and grand means.
    """
    return (
        k
        - k.mean(-2, keepdim=True)
        - k.mean(-1, keepdim=True)
        + k.mean((-2, -1), keepdim=True)
    )


def frobenius(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius product ``Tr[a^T b]`` over the last two dimensions.

    Args:
        a: Matrices of shape ``(..., n, n)``.
        b: Matrices broadcastable against ``a``.

    Returns:
        One product per leading index.
    """
    return (a * b).sum((-2, -1))


def norm(a: torch.Tensor) -> torch.Tensor:
    """Frobenius norm ``||a||_F = sqrt(<a, a>_F)``, Cortes et al. (2012), Section 2.2.

    Args:
        a: Matrices of shape ``(..., n, n)``.

    Returns:
        One norm per leading index.
    """
    return frobenius(a, a).sqrt()


def scale(k_t: torch.Tensor, k_0: torch.Tensor) -> torch.Tensor:
    """Log ratio of kernel Frobenius norms, ``S_t = log(||K_t||_F / ||K_0||_F)``.

    Args:
        k_t: Kernels at step t, shape ``(..., n, n)``.
        k_0: Kernels at initialisation, broadcastable against ``k_t``.

    Returns:
        ``S_t`` per leading index.
    """
    return (norm(k_t) / norm(k_0)).log()


def shape(k_t: torch.Tensor, k_0: torch.Tensor) -> torch.Tensor:
    """One minus the cosine between kernels, ``R_t = 1 - <K_t/||K_t||_F, K_0/||K_0||_F>_F``.

    Computed as half the squared distance between the unit-norm kernels, which
    equals ``R_t`` exactly and does not cancel near zero.

    Args:
        k_t: Kernels at step t, shape ``(..., n, n)``.
        k_0: Kernels at initialisation, broadcastable against ``k_t``.

    Returns:
        ``R_t`` per leading index.
    """
    u_t = k_t / norm(k_t)[..., None, None]
    u_0 = k_0 / norm(k_0)[..., None, None]
    return frobenius(u_t - u_0, u_t - u_0) / 2


def movement(k_t: torch.Tensor, k_0: torch.Tensor) -> torch.Tensor:
    """Relative kernel movement ``D_t = ||K_t - K_0||_F / ||K_0||_F``.

    Args:
        k_t: Kernels at step t, shape ``(..., n, n)``.
        k_0: Kernels at initialisation, broadcastable against ``k_t``.

    Returns:
        ``D_t`` per leading index.
    """
    return norm(k_t - k_0) / norm(k_0)


def alignment(k: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Centred kernel matrix alignment, Cortes et al. (2012), Definition 4.

    Args:
        k: Kernels of shape ``(..., n, n)``.
        g: Target kernel, such as ``Y Y^T``, broadcastable against ``k``.

    Returns:
        ``<K_c, G_c>_F / (||K_c||_F ||G_c||_F)`` per leading index.
    """
    k_c, g_c = centre(k), centre(g)
    return frobenius(k_c, g_c) / (norm(k_c) * norm(g_c))


def uncentred_alignment(k: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Uncentred alignment of Cristianini et al., Cortes et al. (2012), Section 2.3.

    Args:
        k: Kernels of shape ``(..., n, n)``.
        g: Target kernel, such as ``Y Y^T``, broadcastable against ``k``.

    Returns:
        ``<K, G>_F / (||K||_F ||G||_F)`` per leading index.
    """
    return frobenius(k, g) / (norm(k) * norm(g))


def infinite_width(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Gram matrix ``H^inf`` of a bias-free two-layer ReLU network, Basri et al. (2019).

    Their Equation 4 gives the expectation over initialisation of the Gram
    matrix of first-layer gradients, ``x_i^T x_j (pi - arccos(x_i^T x_j)) / (2 pi)``,
    the ``H^inf`` of Arora et al. (2019), Theorem 4.1.

    Args:
        x1: Points on the unit sphere, one per row, with any leading dimensions.
        x2: Points on the unit sphere, broadcastable against ``x1``.

    Returns:
        ``H^inf`` between every row of ``x1`` and every row of ``x2``, with the
        cosines held in ``[-1, 1]``, the domain of ``arccos``, against rounding.
    """
    cosine = (x1 @ x2.mT).clamp(-1, 1)
    return cosine * (torch.pi - cosine.arccos()) / (2 * torch.pi)
