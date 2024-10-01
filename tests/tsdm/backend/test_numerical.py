r"""Tests for the numerical backend."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from tsdm.backend.generic import is_singleton
from tsdm.types.arrays import SupportsShape

CASES: dict[str, tuple[SupportsShape, bool]] = {
    "ndarray-()"       : (np.array(1)                        , True),
    "ndarray-(0,)"     : (np.array([])                       , False),
    "ndarray-(1, 0)"   : (np.array([[]])                     , False),
    "ndarray-(1, 1)"   : (np.array([[1]])                    , True),
    "ndarray-(1,)"     : (np.array([1])                      , True),
    "ndarray-(2, 1)"   : (np.array([[1], [2]])               , False),
    "ndarray-(2,)"     : (np.array([1, 2])                   , False),
    "Series-(0,)"      : (pd.Series([])                      , False),
    "Series-(1,)"      : (pd.Series([1])                     , True),
    "Series-(2,)"      : (pd.Series([1, 2])                  , False),
    "DataFrame-(0, 0)" : (pd.DataFrame([])                   , False),
    "DataFrame-(1, 0)" : (pd.DataFrame([[]])                 , False),
    "DataFrame-(1, 1)" : (pd.DataFrame([[1]])                , True),
    "DataFrame-(1, 2)" : (pd.DataFrame({"x": [1], "y": [2]}) , False),
    "DataFrame-(2, 1)" : (pd.DataFrame([[1], [2]])           , False),
    "DataFrame-(2, 2)" : (pd.DataFrame([[1, 2], [3, 4]])     , False),
    "Table-(0, 0)"     : (pa.table({})                       , False),
    "Table-(0, 1)"     : (pa.table({"x": []})                , False),
    "Table-(1, 1)"     : (pa.table({"x": [1]})               , True),
    "Table-(1, 2)"     : (pa.table({"x": [1], "y": [2]})     , False),
    "Table-(2, 1)"     : (pa.table({"x": [1, 2]})            , False),
    "Tensor-()"        : (torch.tensor(1)                    , True),
    "Tensor-(0,)"      : (torch.tensor([])                   , False),
    "Tensor-(1, 0)"    : (torch.tensor([[]])                 , False),
    "Tensor-(1, 1)"    : (torch.tensor([[1]])                , True),
    "Tensor-(1,)"      : (torch.tensor([1])                  , True),
    "Tensor-(2, 1)"    : (torch.tensor([[1], [2]])           , False),
    "Tensor-(2,)"      : (torch.tensor([1, 2])               , False),
}  # fmt: skip


@pytest.mark.parametrize("name", CASES)
def test_is_singleton(name: str) -> None:
    r"""Test the is_singleton function."""
    x, expected = CASES[name]
    assert is_singleton(x) is expected
    assert name == f"{x.__class__.__name__}-{tuple(x.shape)}"
