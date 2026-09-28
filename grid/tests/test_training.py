# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import functools
from operator import ge, le, lt

from hypothesis import given, reject
from hypothesis import strategies as st
from hypothesis.internal.filtering import max_len, min_len

import deeplearning.training
from deeplearning.training import Cell


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
        alpha=st.floats()
        .filter(functools.partial(le, 2.2227587494850775e-162))
        .filter(functools.partial(ge, 1.3407807929942596e154)),
        chunk_size=st.one_of(
            st.none(), st.none(), st.integers().filter(functools.partial(le, 1))
        ),
        eta_0=st.floats().filter(functools.partial(lt, 0)),
        eta_lambda=st.floats().filter(functools.partial(le, 0)),
        grokked=st.lists(
            st.floats()
            .filter(functools.partial(le, 0))
            .filter(functools.partial(ge, 1))
        )
        .map(tuple)
        .filter(functools.partial(min_len, 1))
        .filter(functools.partial(max_len, 3)),
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
        .filter(functools.partial(max_len, 5)),
        steps=st.integers()
        .filter(functools.partial(le, 1))
        .filter(functools.partial(ge, 100000)),
        train_fraction=st.floats()
        .filter(functools.partial(le, 0))
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
