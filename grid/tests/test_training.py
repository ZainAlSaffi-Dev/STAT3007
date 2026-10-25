# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import functools
from operator import ge, le, lt

from hypothesis import given, reject
from hypothesis import strategies as st
from hypothesis.internal.filtering import max_len, min_len
from torch import Generator

import deeplearning.training
from deeplearning.training import Cell, Setting


@given(
    p=st.integers().filter(functools.partial(le, 1)).filter(functools.partial(ge, 23)),
    train_fraction=st.floats()
    .filter(functools.partial(le, 0))
    .filter(functools.partial(ge, 1)),
    width=st.integers()
    .filter(functools.partial(le, 1))
    .filter(functools.partial(ge, 100)),
    parameterisation=st.sampled_from(["mean_field", "ntk"]),
    settings=st.lists(
        st.builds(
            Setting,
            alpha=st.floats()
            .filter(functools.partial(le, 0.5))
            .filter(functools.partial(ge, 2.0)),
            eta_0=st.floats()
            .filter(functools.partial(lt, 0))
            .filter(functools.partial(ge, 100.0)),
            eta_lambda=st.floats()
            .filter(functools.partial(le, 0))
            .filter(functools.partial(ge, 0.2)),
        )
    )
    .map(tuple)
    .filter(functools.partial(min_len, 1)),
    steps=st.integers()
    .filter(functools.partial(le, 1))
    .filter(functools.partial(ge, 100000)),
    seeds=st.lists(
        st.integers()
        .filter(functools.partial(le, -2_147_483_648))
        .filter(functools.partial(ge, 2_147_483_647))
    )
    .map(tuple)
    .filter(functools.partial(min_len, 1))
    .filter(functools.partial(max_len, 12)),
    memorised=st.floats()
    .filter(functools.partial(le, 0))
    .filter(functools.partial(ge, 1)),
    grokked=st.lists(
        st.floats().filter(functools.partial(le, 0)).filter(functools.partial(ge, 1))
    )
    .map(tuple)
    .filter(functools.partial(min_len, 1))
    .filter(functools.partial(max_len, 3)),
    threshold=st.floats()
    .filter(functools.partial(le, 0))
    .filter(functools.partial(ge, 1)),
    kernels=st.booleans(),
)
def test_fuzz_Cell(
    p,
    train_fraction,
    width,
    parameterisation,
    settings,
    steps,
    seeds,
    memorised,
    grokked,
    threshold,
    kernels,
):
    try:
        deeplearning.training.Cell(
            p=p,
            train_fraction=train_fraction,
            width=width,
            parameterisation=parameterisation,
            settings=settings,
            steps=steps,
            seeds=seeds,
            memorised=memorised,
            grokked=grokked,
            threshold=threshold,
            kernels=kernels,
        )
    except ValueError:
        reject()


@given(
    alpha=st.floats()
    .filter(functools.partial(le, 0.5))
    .filter(functools.partial(ge, 2.0)),
    eta_0=st.floats()
    .filter(functools.partial(lt, 0))
    .filter(functools.partial(ge, 100.0)),
    eta_lambda=st.floats()
    .filter(functools.partial(le, 0))
    .filter(functools.partial(ge, 0.2)),
)
def test_fuzz_Setting(alpha, eta_0, eta_lambda):
    try:
        deeplearning.training.Setting(alpha=alpha, eta_0=eta_0, eta_lambda=eta_lambda)
    except ValueError:
        reject()


@given(
    inputs=st.integers()
    .filter(functools.partial(le, 1))
    .filter(functools.partial(ge, 46)),
    width=st.integers()
    .filter(functools.partial(le, 1))
    .filter(functools.partial(ge, 100)),
    outputs=st.integers()
    .filter(functools.partial(le, 1))
    .filter(functools.partial(ge, 23)),
    parameterisation=st.sampled_from(["mean_field", "ntk"]),
    generator=st.builds(Generator),
)
def test_fuzz_TwoLayer(inputs, width, outputs, parameterisation, generator):
    try:
        deeplearning.training.TwoLayer(
            inputs=inputs,
            width=width,
            outputs=outputs,
            parameterisation=parameterisation,
            generator=generator,
        )
    except ValueError:
        reject()


@given(
    steps=st.integers()
    .filter(functools.partial(le, 1))
    .filter(functools.partial(ge, 9_007_199_254_740_992))
)
def test_fuzz_checkpoints(steps):
    try:
        deeplearning.training.checkpoints(steps=steps)
    except ValueError:
        reject()


@given(
    p=st.integers().filter(functools.partial(le, 1)).filter(functools.partial(ge, 23)),
    train_fraction=st.floats()
    .filter(functools.partial(le, 0))
    .filter(functools.partial(ge, 1)),
    seed=st.integers()
    .filter(functools.partial(le, -2_147_483_648))
    .filter(functools.partial(ge, 2_147_483_647)),
)
def test_fuzz_modular_addition(p, train_fraction, seed):
    try:
        deeplearning.training.modular_addition(
            p=p, train_fraction=train_fraction, seed=seed
        )
    except ValueError:
        reject()


@given(
    cell=st.builds(
        Cell,
        grokked=st.lists(
            st.floats()
            .filter(functools.partial(le, 0))
            .filter(functools.partial(ge, 1))
        )
        .map(tuple)
        .filter(functools.partial(min_len, 1))
        .filter(functools.partial(max_len, 3)),
        kernels=st.one_of(st.just(True), st.booleans()),
        memorised=st.floats()
        .filter(functools.partial(le, 0))
        .filter(functools.partial(ge, 1)),
        p=st.integers()
        .filter(functools.partial(le, 1))
        .filter(functools.partial(ge, 23)),
        parameterisation=st.sampled_from(["mean_field", "ntk"]),
        seeds=st.lists(
            st.integers()
            .filter(functools.partial(le, -2_147_483_648))
            .filter(functools.partial(ge, 2_147_483_647))
        )
        .map(tuple)
        .filter(functools.partial(min_len, 1))
        .filter(functools.partial(max_len, 12)),
        settings=st.lists(
            st.builds(
                Setting,
                alpha=st.floats()
                .filter(functools.partial(le, 0.5))
                .filter(functools.partial(ge, 2.0)),
                eta_0=st.floats()
                .filter(functools.partial(lt, 0))
                .filter(functools.partial(ge, 100.0)),
                eta_lambda=st.floats()
                .filter(functools.partial(le, 0))
                .filter(functools.partial(ge, 0.2)),
            )
        )
        .map(tuple)
        .filter(functools.partial(min_len, 1))
        .filter(functools.partial(max_len, 4)),
        steps=st.integers()
        .filter(functools.partial(le, 1))
        .filter(functools.partial(ge, 100000)),
        threshold=st.floats()
        .filter(functools.partial(le, 0))
        .filter(functools.partial(ge, 1)),
        train_fraction=st.floats()
        .filter(functools.partial(le, 0.9))
        .filter(functools.partial(ge, 1)),
        width=st.integers()
        .filter(functools.partial(le, 1))
        .filter(functools.partial(ge, 100)),
    )
)
def test_fuzz_train(cell):
    try:
        deeplearning.training.train(cell=cell)
    except ValueError:
        reject()
