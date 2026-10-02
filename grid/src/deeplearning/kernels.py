"""Empirical neural tangent kernels and the scale, shape and alignment of their movement.

The kernel is the Gram matrix of parameter gradients on a fixed probe set. Two
computations follow the torch.func neural tangent kernel tutorial: Jacobian
contraction and NTK-vector products, sections 3.2 and 3.3 of Novak et al.
(2022). For a network with one output, the tutorial's ``trace`` contraction
gives the ``n x n`` kernel.

Centring, the Frobenius product and centred alignment follow Cortes, Mohri and
Rostamizadeh (2012): Equation 1, Lemma 1 and Definition 4.

The infinite-width Gram matrix of a bias-free two-layer ReLU network on the
sphere is Equation 4 of Basri et al. (2019).

Every function takes kernels with any leading batch dimensions, so one call
covers all seeds and checkpoints of a cell.

A collapsing kernel stays measurable. The Frobenius norm scales each kernel
before squaring, the two-pass algorithm listed by Higham (2002), Section 27.8,
and timed by Anderson (2017), Section 3.2, since a plain sum of squares
underflows once every entry is below the square root of the smallest normal
number (Blue 1978), although S_t, R_t and A_t are still representable, and
reliable software returns them accurately (Demmel 1984). The scale is the
largest power of 2 not above the largest entry, which divides without error
(Blue 1978, Lemma A), as Higham advises to avoid the extra rounding of the
scaled evaluation. The inner products are taken between kernels divided by
that power and then by the norm of the result, so no subnormal norm is formed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import icontract
import numpy as np
import scipy.linalg
import torch
from torch.func import jacrev, jvp, vjp, vmap

Params = Mapping[str, torch.Tensor]
SingleInput = Callable[[Params, torch.Tensor], torch.Tensor]


def nrm2(a: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of each matrix by BLAS nrm2, the postconditions' reference.

    ``scipy.linalg.norm`` of a vector calls BLAS nrm2, which the reference BLAS
    computes with safe scaling (Anderson 2017, Section 1.1).

    Args:
        a: Matrices of shape ``(..., n, n)``.

    Returns:
        One norm per leading index.
    """
    return torch.as_tensor(
        np.apply_along_axis(scipy.linalg.norm, -1, a.flatten(-2).numpy(force=True)),
        device=a.device,
    )


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


@icontract.ensure(
    lambda fnet_single, params, x1, x2, compute, result: torch.allclose(
        result,
        entk_jacobian_contraction(fnet_single, params, x1, x2, compute),
        atol=1e-5,
    ),
    "NTK-vector products and Jacobian contraction give the same kernel",
    enabled=icontract.SLOW,
)
def entk_ntk_vps(
    fnet_single: SingleInput,
    params: Params,
    x1: torch.Tensor,
    x2: torch.Tensor,
    compute: str = "full",
) -> torch.Tensor:
    """Empirical NTK between ``x1`` and ``x2`` from NTK-vector products.

    The postcondition checks the kernel against ``entk_jacobian_contraction``
    with the tolerance of the tutorial's own comparison of the two methods.

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
        """NTK between a single pair of data points.

        Args:
            x_1: One data point.
            x_2: The other data point.

        Returns:
            The ``O x O`` kernel block, one NTK-vector product per column.
        """

        def func_x1(p: Params) -> torch.Tensor:
            """Network at ``x_1`` as a function of its parameters.

            Args:
                p: Parameters.

            Returns:
                The output at ``x_1``.
            """
            return fnet_single(p, x_1)

        def func_x2(p: Params) -> torch.Tensor:
            """Network at ``x_2`` as a function of its parameters.

            Args:
                p: Parameters.

            Returns:
                The output at ``x_2``.
            """
            return fnet_single(p, x_2)

        output, vjp_fn = vjp(func_x1, params)[:2]

        def get_ntk_slice(vec: torch.Tensor) -> torch.Tensor:
            """NTK-vector product ``J(x_2) J(x_1)^T vec`` for one basis vector.

            Args:
                vec: A column of the identity matrix.

            Returns:
                The kernel block applied to ``vec``.
            """
            vjps = vjp_fn(vec)
            jvps: torch.Tensor = jvp(func_x2, (params,), vjps)[1]
            return jvps

        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device)
        ntk: torch.Tensor = vmap(get_ntk_slice)(basis)
        return ntk

    full = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    return torch.einsum(BLOCKS[compute], full)


@icontract.ensure(
    # assert_close raises with the mismatched count and the greatest differences
    # and their indices, which icontract cannot recompute for this condition.
    lambda w1, w2, x, u, hidden, readout, result: (
        torch.testing.assert_close(
            result,
            vmap(
                lambda prm: entk_jacobian_contraction(
                    lambda p, xi: (
                        readout
                        * (u @ (p["w2"] @ torch.relu(hidden * (xi @ p["w1"].mT))))
                    )[None],
                    prm,
                    x,
                    x,
                    "trace",
                )
            )(
                {
                    "w1": w1.reshape(-1, *w1.shape[-2:]),
                    "w2": w2.reshape(-1, *w2.shape[-2:]),
                }
            ).reshape(result.shape),
        )
        is None
    ),
    "the closed form is the Jacobian contraction of u^T f",
    enabled=icontract.SLOW,
)
def two_layer_entk(
    w1: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    hidden: float,
    readout: float,
) -> torch.Tensor:
    """Empirical NTK of ``u^T f`` for ``f(x) = c W2 relu(s W1 x)``, in closed form.

    A fully connected layer's contribution to the NTK is the inner product of
    its inputs times that of its output cotangents, the structured derivatives
    of Novak et al. (2022), Section 3.4, worked for a fully connected layer in
    their Section 4.1. For this network the second layer contributes
    ``c^2 (u^T u) r^T r'`` and the first ``c^2 s^2 (x^T x') (delta^T delta')``
    with ``r = relu(z)``, ``z = s W1 x`` and ``delta = (W2^T u) * relu'(z)``,
    where ``relu'`` is ``1`` above zero and ``0`` elsewhere, the minimum-norm
    subgradient autograd takes at the kink (PyTorch, Autograd mechanics,
    Gradients for non-differentiable functions). Their Neural Tangents library
    implements the method only in JAX and requires TensorFlow, so the formula is
    written here and the postcondition checks it against the Jacobian
    contraction.

    The kernel jumps where a preactivation crosses zero, and mathematically
    identical products need not round alike (PyTorch, Numerical accuracy), so
    ``z`` is computed as the network's forward computes it: ``W1 x`` is the
    matrix product with the one-hot inputs (Gromov, 2023, Eq. 1), then scaled
    by ``s``. A preactivation within rounding of zero then falls on the same
    side as in training.

    Args:
        w1: First-layer weights ``(..., N, D)``, one matrix per leading index.
        w2: Readout weights ``(..., O, N)``, with the leading shape of ``w1``.
        x: One-hot inputs ``(n, D)``, one example per row.
        u: Weights of the scalar output ``u^T f`` over the ``O`` logits.
        hidden: Hidden scale ``s``.
        readout: Readout scale ``c``.

    Returns:
        The ``n x n`` kernel per leading index.
    """
    z = hidden * (x @ w1.mT)
    r = z.relu()
    delta = (z > 0).to(z.dtype) * (u @ w2)[..., None, :]
    return readout**2 * (
        (u @ u) * r @ r.mT + hidden**2 * (x @ x.mT) * (delta @ delta.mT)
    )


@icontract.ensure(
    lambda result: torch.allclose(result.sum(-1), torch.zeros_like(result.sum(-1))),
    "every row sums to zero, since H 1 = 0",
    enabled=icontract.SLOW,
)
@icontract.ensure(
    lambda result: torch.allclose(result.sum(-2), torch.zeros_like(result.sum(-2))),
    "every column sums to zero, since 1^T H = 0",
    enabled=icontract.SLOW,
)
def centre(k: torch.Tensor) -> torch.Tensor:
    """Centre kernel matrices in feature space, Cortes et al. (2012), Equation 1.

    By Lemma 1.1 the result is ``H k H`` with ``H 1 = 0``, so every row and
    column of every centred kernel sums to zero, which the postconditions check.

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


@icontract.ensure(
    lambda a, b, result: torch.allclose(
        result, torch.linalg.vecdot(a.flatten(-2), b.flatten(-2)), equal_nan=True
    ),
    "Tr[a^T b] is the dot product of the flattened matrices",
    enabled=icontract.SLOW,
)
def frobenius(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius product ``Tr[a^T b]`` over the last two dimensions.

    The postcondition checks it against ``torch.linalg.vecdot`` of the
    flattened matrices.

    Args:
        a: Matrices of shape ``(..., n, n)``.
        b: Matrices broadcastable against ``a``.

    Returns:
        One product per leading index.
    """
    return (a * b).sum((-2, -1))


@icontract.ensure(
    lambda result: torch.equal(
        torch.frexp(result).mantissa, torch.full_like(result, 0.5)
    ),
    "the result is a power of 2",
    enabled=icontract.SLOW,
)
@icontract.ensure(
    lambda a, result: bool(
        (
            ((m := a.abs().amax((-2, -1), keepdim=True)) < 2 * result)
            & ((result <= m) | (m == 0))
        ).all()
    ),
    "the result is the largest power of 2 not above max |a_ij|",
    enabled=icontract.SLOW,
)
def base_power(a: torch.Tensor) -> torch.Tensor:
    """Largest power of 2 not above ``max |a_ij|``, the scale of each matrix.

    Dividing by a power of the base is exact while the quotient stays in
    range (Blue 1978, Lemma A), and this power keeps the largest quotient in
    ``[1, 2)``. It is formed as ``1 * 2^(e - 1)`` from the exponent ``e`` of
    ``torch.frexp``, a representable number for every finite nonzero maximum,
    subnormal included, where ``torch.ldexp(a, -e)`` would multiply by an
    overflowing ``2^-e``. A zero matrix gets ``1/2``.

    Args:
        a: Matrices of shape ``(..., n, n)``.

    Returns:
        One power of 2 per leading index, shaped ``(..., 1, 1)``.
    """
    w = a.abs().amax((-2, -1), keepdim=True)
    return torch.ldexp(torch.ones_like(w), torch.frexp(w).exponent - 1)


@icontract.ensure(
    lambda a, result: torch.allclose(result, nrm2(a), atol=0, equal_nan=True),
    "the Frobenius norm of BLAS nrm2, relative to its size",
    enabled=icontract.SLOW,
)
def norm(a: torch.Tensor) -> torch.Tensor:
    """Frobenius norm ``||a||_F = sqrt(<a, a>_F)``, Cortes et al. (2012), Section 2.2.

    Computed as ``w sqrt(sum (a_ij / w)^2)`` with ``w`` from ``base_power``,
    the two-pass algorithm of Higham (2002), Section 27.8, so it underflows
    only where the norm itself does. The postcondition checks it against BLAS
    nrm2 with no absolute tolerance, which would pass any two tiny norms.

    Args:
        a: Matrices of shape ``(..., n, n)``.

    Returns:
        One norm per leading index.
    """
    w = base_power(a)
    return (w * (a / w).square().sum((-2, -1), keepdim=True).sqrt()).squeeze((-2, -1))


@icontract.ensure(
    lambda a, result: torch.allclose(
        frobenius(result, result), nrm2(a) / nrm2(a), equal_nan=True
    ),
    "the result has unit Frobenius norm, <u, u>_F = 1",
    enabled=icontract.SLOW,
)
@icontract.ensure(
    lambda a, result: torch.allclose(
        torch.nn.functional.cosine_similarity(
            result.flatten(-2), a.flatten(-2) / nrm2(a)[..., None], dim=-1, eps=0
        ),
        nrm2(a) / nrm2(a),
        equal_nan=True,
    ),
    "the result points along a",
    enabled=icontract.SLOW,
)
def unit(a: torch.Tensor) -> torch.Tensor:
    """Each matrix divided by its Frobenius norm, ``a / ||a||_F``.

    Divides ``a / w`` by its norm, with ``w`` from ``base_power``, rather than
    ``a`` by ``||a||_F``: where ``a`` is subnormal its norm keeps only a few
    bits, while ``a / w`` is normal and exact. This is the scaling of Higham
    (2002), Section 27.8, carried into the normalisation.

    Args:
        a: Matrices of shape ``(..., n, n)``.

    Returns:
        The unit matrices, ``NaN`` for a zero matrix.
    """
    scaled = a / base_power(a)
    return scaled / norm(scaled)[..., None, None]


@icontract.ensure(
    lambda k_t, k_0, result: torch.allclose(
        result,
        1
        - torch.nn.functional.cosine_similarity(
            k_t.flatten(-2) / nrm2(k_t)[..., None],
            k_0.flatten(-2) / nrm2(k_0)[..., None],
            dim=-1,
            eps=0,
        ),
        equal_nan=True,
    ),
    "R_t is one minus the cosine similarity of the flattened unit kernels",
    enabled=icontract.SLOW,
)
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
    difference = unit(k_t) - unit(k_0)
    return frobenius(difference, difference) / 2


@icontract.ensure(
    lambda k_t, k_0, result: torch.allclose(
        result, nrm2(k_t - k_0) / nrm2(k_0), equal_nan=True
    ),
    "D_t is the ratio of BLAS nrm2 norms",
    enabled=icontract.SLOW,
)
def movement(k_t: torch.Tensor, k_0: torch.Tensor) -> torch.Tensor:
    """Relative kernel movement ``D_t = ||K_t - K_0||_F / ||K_0||_F``.

    Args:
        k_t: Kernels at step t, shape ``(..., n, n)``.
        k_0: Kernels at initialisation, broadcastable against ``k_t``.

    Returns:
        ``D_t`` per leading index.
    """
    return norm(k_t - k_0) / norm(k_0)


@icontract.ensure(
    lambda k, g, result: torch.allclose(
        result,
        torch.nn.functional.cosine_similarity(
            centre(k).flatten(-2) / nrm2(centre(k))[..., None],
            centre(g).flatten(-2) / nrm2(centre(g))[..., None],
            dim=-1,
            eps=0,
        ),
        equal_nan=True,
    ),
    "the alignment is the cosine similarity of the flattened centred unit kernels",
    enabled=icontract.SLOW,
)
def alignment(k: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Centred kernel matrix alignment, Cortes et al. (2012), Definition 4.

    The centred kernels are divided by their norms before the product, so it
    does not underflow.

    Args:
        k: Kernels of shape ``(..., n, n)``.
        g: Target kernel, such as ``Y Y^T``, broadcastable against ``k``.

    Returns:
        ``<K_c, G_c>_F / (||K_c||_F ||G_c||_F)`` per leading index.
    """
    return frobenius(unit(centre(k)), unit(centre(g)))


@icontract.ensure(
    lambda x1, x2, result: torch.allclose(
        result,
        (x1 @ x2.mT).clamp(-1, 1)
        * (-(x1 @ x2.mT)).clamp(-1, 1).arccos()
        / (2 * torch.pi),
    ),
    "pi - arccos(c) = arccos(-c), DLMF 4.23.11",
)
def infinite_width(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Gram matrix ``H^inf`` of a bias-free two-layer ReLU network, Basri et al. (2019).

    Their Equation 4 gives the expectation over initialisation of the Gram
    matrix of first-layer gradients, ``x_i^T x_j (pi - arccos(x_i^T x_j)) / (2 pi)``,
    the ``H^inf`` of Arora et al. (2019), Theorem 4.1. The postcondition
    restates it through the reflection ``arccos(-c) = pi - arccos(c)`` of
    DLMF 4.23.11.

    Args:
        x1: Points on the unit sphere, one per row, with any leading dimensions.
        x2: Points on the unit sphere, broadcastable against ``x1``.

    Returns:
        ``H^inf`` between every row of ``x1`` and every row of ``x2``, with the
        cosines held in ``[-1, 1]``, the domain of ``arccos``, against rounding.
    """
    cosine = (x1 @ x2.mT).clamp(-1, 1)
    return cosine * (torch.pi - cosine.arccos()) / (2 * torch.pi)
