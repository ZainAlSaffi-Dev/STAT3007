from __future__ import annotations

import importlib.metadata

import deeplearning as m


def test_version() -> None:
    assert importlib.metadata.version("deeplearning") == m.__version__
