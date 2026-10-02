# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

import deeplearning.survival

UNIT = {"min_value": -1, "max_value": 1}
study = st.shared(array_shapes(min_dims=4, max_dims=4), key="study")
covariates = study.flatmap(
    lambda s: arrays(np.float64, (s[0], s[1]), elements=UNIT)
).map(torch.from_numpy)
times = study.flatmap(
    lambda s: arrays(np.int64, s[0], elements=st.integers(min_value=1, max_value=s[0]))
).map(torch.from_numpy)
observations = study.flatmap(lambda s: arrays(np.bool_, s[0])).map(torch.from_numpy)
clusters = study.flatmap(
    lambda s: arrays(
        np.int64, s[0], elements=st.integers(min_value=0, max_value=s[2] - 1)
    )
).map(torch.from_numpy)
draws = study.flatmap(
    lambda s: arrays(
        np.int64, (s[3], s[2]), elements=st.integers(min_value=0, max_value=s[2])
    )
).map(lambda a: torch.from_numpy(a).to(torch.float64))


@given(
    covariates=covariates,
    time=times,
    observed=observations,
    cluster=clusters,
    counts=draws,
)
def test_fuzz_cumulative_regression(
    covariates: torch.Tensor,
    time: torch.Tensor,
    observed: torch.Tensor,
    cluster: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    deeplearning.survival.cumulative_regression(
        covariates=covariates,
        time=time,
        observed=observed,
        cluster=cluster,
        counts=counts,
    )
