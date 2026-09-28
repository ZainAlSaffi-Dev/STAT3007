# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings, target
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from torch.func import functional_call

import deeplearning.kernels
from deeplearning.training import TwoLayer

FINFO = np.finfo(np.float64)
UNIT = {"min_value": -1, "max_value": 1}
factor_shape = st.shared(array_shapes(min_dims=2, max_dims=2, min_side=2), key="shape")
kernels_psd = (
    arrays(np.int8, factor_shape)
    .filter(lambda b: bool((b != b[0]).any()))
    .map(lambda b: torch.from_numpy(b.astype(np.float64) @ b.T.astype(np.float64)))
)
scales = st.floats(min_value=FINFO.eps, max_value=1 / FINFO.eps)
models = st.shared(
    st.builds(
        lambda dims, parameterisation, seed: TwoLayer(
            dims[0],
            dims[1],
            dims[2],
            parameterisation,
            torch.Generator().manual_seed(seed),
        ),
        dims=array_shapes(min_dims=3, max_dims=3),
        parameterisation=st.sampled_from(["ntk", "mean_field"]),
        seed=st.integers(min_value=-(2**31), max_value=2**31 - 1),
    ),
    key="model",
)
probes = array_shapes(min_dims=1, max_dims=1)


def cortes_centre(k: torch.Tensor) -> torch.Tensor:
    """Cortes et al. (2012), Lemma 1.1: ``[I - 11^T/m] K [I - 11^T/m]``."""
    m = k.shape[-1]
    h = torch.eye(m, dtype=k.dtype) - torch.ones(m, m, dtype=k.dtype) / m
    return h @ k @ h


def cortes_alignment(k: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Cortes et al. (2012), Definition 4, with ``<A, B>_F = Tr[A^T B]``."""
    k_c, g_c = cortes_centre(k), cortes_centre(g)
    return torch.trace(k_c.T @ g_c) / (
        torch.sqrt(torch.trace(k_c.T @ k_c)) * torch.sqrt(torch.trace(g_c.T @ g_c))
    )


@pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated:FutureWarning")
@settings(deadline=None)
@given(
    compute=st.sampled_from(["full", "trace", "diagonal"]),
    fnet_single=models.map(
        lambda m: (
            lambda prm, x: functional_call(m, (dict(prm),), (x.unsqueeze(0),)).squeeze(
                0
            )
        )
    ),
    params=models.map(lambda m: {k: v.detach() for k, v in m.named_parameters()}),
    x1=st.tuples(models, probes)
    .flatmap(
        lambda mp: arrays(np.float64, (mp[1][0], mp[0].w1.shape[1]), elements=UNIT)
    )
    .map(torch.from_numpy),
    x2=st.tuples(models, probes)
    .flatmap(
        lambda mp: arrays(np.float64, (mp[1][0], mp[0].w1.shape[1]), elements=UNIT)
    )
    .map(torch.from_numpy),
)
def test_equivalent_entk_jacobian_contraction_entk_ntk_vps(
    compute, fnet_single, params, x1, x2
) -> None:
    result_entk_jacobian_contraction = deeplearning.kernels.entk_jacobian_contraction(
        fnet_single=fnet_single, params=params, x1=x1, x2=x2, compute=compute
    )
    result_entk_ntk_vps = deeplearning.kernels.entk_ntk_vps(
        fnet_single=fnet_single, params=params, x1=x1, x2=x2, compute=compute
    )
    torch.testing.assert_close(result_entk_jacobian_contraction, result_entk_ntk_vps)


@given(k=kernels_psd)
def test_idempotent_centre(k: torch.Tensor) -> None:
    result = deeplearning.kernels.centre(k=k)
    repeat = deeplearning.kernels.centre(k=result)
    torch.testing.assert_close(result, repeat)


@given(k=kernels_psd)
def test_equivalent_centre_cortes_lemma_1(k: torch.Tensor) -> None:
    torch.testing.assert_close(deeplearning.kernels.centre(k), cortes_centre(k))


@given(k=kernels_psd, g=kernels_psd)
def test_equivalent_alignment_cortes_definition_4(
    k: torch.Tensor, g: torch.Tensor
) -> None:
    torch.testing.assert_close(
        deeplearning.kernels.alignment(k, g), cortes_alignment(k, g)
    )


@given(k=kernels_psd, g=kernels_psd, c=scales)
def test_alignment_invariant_to_scale(
    k: torch.Tensor, g: torch.Tensor, c: float
) -> None:
    before, after = (
        deeplearning.kernels.alignment(k, g),
        deeplearning.kernels.alignment(c * k, g),
    )
    target(abs(float(after - before)))
    torch.testing.assert_close(after, before)


@given(k_t=kernels_psd, k_0=kernels_psd, c=scales)
def test_shape_invariant_to_scale(
    k_t: torch.Tensor, k_0: torch.Tensor, c: float
) -> None:
    before, after = (
        deeplearning.kernels.shape(k_t, k_0),
        deeplearning.kernels.shape(c * k_t, k_0),
    )
    target(abs(float(after - before)))
    torch.testing.assert_close(after, before)


@given(k_t=kernels_psd, k_0=kernels_psd, c=scales)
def test_scale_shifts_by_log_c(k_t: torch.Tensor, k_0: torch.Tensor, c: float) -> None:
    shifted = deeplearning.kernels.scale(c * k_t, k_0)
    expected = deeplearning.kernels.scale(k_t, k_0) + math.log(c)
    target(abs(float(shifted - expected)))
    torch.testing.assert_close(shifted, expected)


@given(k_t=kernels_psd, k_0=kernels_psd)
def test_shape_in_unit_interval_for_psd(k_t: torch.Tensor, k_0: torch.Tensor) -> None:
    r = deeplearning.kernels.shape(
        deeplearning.kernels.centre(k_t), deeplearning.kernels.centre(k_0)
    )
    target(float(r))
    torch.testing.assert_close(r.clamp(0, 1), r)


@given(k_t=kernels_psd, k_0=kernels_psd)
def test_movement_decomposes_into_scale_and_shape(
    k_t: torch.Tensor, k_0: torch.Tensor
) -> None:
    s, r = deeplearning.kernels.scale(k_t, k_0), deeplearning.kernels.shape(k_t, k_0)
    d = deeplearning.kernels.movement(k_t, k_0)
    target(float(d))
    torch.testing.assert_close(d**2, torch.exp(2 * s) + 1 - 2 * torch.exp(s) * (1 - r))
