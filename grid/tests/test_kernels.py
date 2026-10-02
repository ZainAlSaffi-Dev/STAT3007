# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import collections.abc

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from torch.func import functional_call

import deeplearning.kernels
from deeplearning.training import Seed, TwoLayer

UNIT = {"min_value": -1, "max_value": 1}
factor_shape = st.shared(array_shapes(min_dims=2, min_side=2), key="shape")
kernels = st.builds(
    torch.mul,
    arrays(np.int8, factor_shape)
    .map(lambda b: torch.from_numpy(b.astype(np.float64)))
    .map(lambda b: b @ b.mT),
    st.floats(min_value=0, max_value=1, exclude_min=True),
)
spheres = (
    arrays(np.float64, factor_shape, elements=UNIT)
    .filter(lambda x: bool((x != 0).any(axis=-1).all()))
    .map(lambda x: torch.nn.functional.normalize(torch.from_numpy(x), dim=-1))
)
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
        seed=st.from_type(Seed),
    ),
    key="model",
)
fnet_singles = models.map(
    lambda m: (
        lambda prm, x: functional_call(m, (dict(prm),), (x.unsqueeze(0),)).squeeze(0)
    )
)
params = models.map(lambda m: {k: v.detach() for k, v in m.named_parameters()})
probes = st.tuples(models, array_shapes(min_dims=1, max_dims=1)).flatmap(
    lambda mp: arrays(
        np.int64,
        (mp[1][0], 2),
        elements=st.integers(min_value=0, max_value=mp[0].w1.shape[1] - 1),
    ).map(
        lambda t: (
            torch.nn.functional.one_hot(torch.from_numpy(t), mp[0].w1.shape[1])
            .sum(-2)
            .to(torch.float64)
        )
    )
)
computes = st.sampled_from(sorted(deeplearning.kernels.CONTRACTIONS))
network = st.shared(
    st.tuples(
        array_shapes(min_dims=0, max_dims=2), array_shapes(min_dims=3, max_dims=3)
    ),
    key="network",
)
first_layers = network.flatmap(
    lambda s: arrays(np.float64, (*s[0], s[1][0], s[1][1]), elements=UNIT)
).map(torch.from_numpy)
readouts = network.flatmap(
    lambda s: arrays(np.float64, (*s[0], s[1][2], s[1][0]), elements=UNIT)
).map(torch.from_numpy)
inputs = st.tuples(network, array_shapes(min_dims=1, max_dims=1)).flatmap(
    lambda t: arrays(
        np.int64,
        (t[1][0], 2),
        elements=st.integers(min_value=0, max_value=t[0][1][1] - 1),
    ).map(
        lambda h: (
            torch.nn.functional.one_hot(torch.from_numpy(h), t[0][1][1])
            .sum(-2)
            .to(torch.float64)
        )
    )
)
logit_weights = network.flatmap(
    lambda s: arrays(np.float64, (s[1][2],), elements=UNIT)
).map(torch.from_numpy)
layer_scales = st.floats(min_value=0, max_value=1, exclude_min=True)


@given(fnet_single=fnet_singles, params=params, x1=probes, x2=probes, compute=computes)
def test_fuzz_entk_jacobian_contraction(
    fnet_single: collections.abc.Callable[
        [collections.abc.Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor
    ],
    params: collections.abc.Mapping[str, torch.Tensor],
    x1: torch.Tensor,
    x2: torch.Tensor,
    compute: str,
) -> None:
    deeplearning.kernels.entk_jacobian_contraction(
        fnet_single=fnet_single, params=params, x1=x1, x2=x2, compute=compute
    )


@given(fnet_single=fnet_singles, params=params, x1=probes, x2=probes, compute=computes)
def test_fuzz_entk_ntk_vps(
    fnet_single: collections.abc.Callable[
        [collections.abc.Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor
    ],
    params: collections.abc.Mapping[str, torch.Tensor],
    x1: torch.Tensor,
    x2: torch.Tensor,
    compute: str,
) -> None:
    deeplearning.kernels.entk_ntk_vps(
        fnet_single=fnet_single, params=params, x1=x1, x2=x2, compute=compute
    )


@given(
    w1=first_layers,
    w2=readouts,
    x=inputs,
    u=logit_weights,
    hidden=layer_scales,
    readout=layer_scales,
)
def test_fuzz_two_layer_entk(
    w1: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    hidden: float,
    readout: float,
) -> None:
    deeplearning.kernels.two_layer_entk(
        w1=w1, w2=w2, x=x, u=u, hidden=hidden, readout=readout
    )


@given(k=kernels)
def test_fuzz_centre(k: torch.Tensor) -> None:
    deeplearning.kernels.centre(k=k)


@given(a=kernels, b=kernels)
def test_fuzz_frobenius(a: torch.Tensor, b: torch.Tensor) -> None:
    deeplearning.kernels.frobenius(a=a, b=b)


@given(a=kernels)
def test_fuzz_base_power(a: torch.Tensor) -> None:
    deeplearning.kernels.base_power(a=a)


@given(a=kernels)
def test_fuzz_nrm2(a: torch.Tensor) -> None:
    deeplearning.kernels.nrm2(a=a)


@given(a=kernels)
def test_fuzz_norm(a: torch.Tensor) -> None:
    deeplearning.kernels.norm(a=a)


@given(a=kernels)
def test_fuzz_unit(a: torch.Tensor) -> None:
    deeplearning.kernels.unit(a=a)


@given(k_t=kernels, k_0=kernels)
def test_fuzz_shape(k_t: torch.Tensor, k_0: torch.Tensor) -> None:
    deeplearning.kernels.shape(k_t=k_t, k_0=k_0)


@given(k_t=kernels, k_0=kernels)
def test_fuzz_movement(k_t: torch.Tensor, k_0: torch.Tensor) -> None:
    deeplearning.kernels.movement(k_t=k_t, k_0=k_0)


@given(k=kernels, g=kernels)
def test_fuzz_alignment(k: torch.Tensor, g: torch.Tensor) -> None:
    deeplearning.kernels.alignment(k=k, g=g)


@given(x1=spheres, x2=spheres)
def test_fuzz_infinite_width(x1: torch.Tensor, x2: torch.Tensor) -> None:
    deeplearning.kernels.infinite_width(x1=x1, x2=x2)
