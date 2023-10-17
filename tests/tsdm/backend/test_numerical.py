"""Tests for the numerical backend."""

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from tsdm.backend import is_singleton


def test_is_singleton() -> None:
    """Test the is_singleton function."""
    # numpy
    assert is_singleton(np.array(1))
    assert is_singleton(np.array([1]))
    assert is_singleton(np.array([[1]]))
    assert not is_singleton(np.array([]))
    assert not is_singleton(np.array([1, 2]))
    assert not is_singleton(np.array([[1], [2]]))

    # pandas.Series
    assert is_singleton(pd.Series(1))
    assert is_singleton(pd.Series([1]))
    assert is_singleton(pd.Series([[1]]))
    assert not is_singleton(pd.Series([1, 2]))
    assert not is_singleton(pd.Series([]))

    # pandas.DataFrame
    assert is_singleton(pd.DataFrame([1]))
    assert is_singleton(pd.DataFrame([[1]]))
    assert not is_singleton(pd.DataFrame([1, 2]))
    assert not is_singleton(pd.DataFrame([]))
    assert not is_singleton(pd.DataFrame([[1], [2]]))
    assert not is_singleton(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    # pyarrow.Table
    assert is_singleton(pa.table({"x": [1]}))
    assert is_singleton(pa.table({"x": [[1]]}))
    assert not is_singleton(pa.table({"x": [1, 2]}))
    assert not is_singleton(pa.table({"x": [1], "y": [2]}))

    # torch.Tensor
    assert is_singleton(torch.tensor(1))
    assert is_singleton(torch.tensor([1]))
    assert is_singleton(torch.tensor([[1]]))
    assert not is_singleton(torch.tensor([]))
    assert not is_singleton(torch.tensor([1, 2]))
    assert not is_singleton(torch.tensor([[1], [2]]))

    # assert is_singleton(1)
    # assert is_singleton(1.0)
    # assert is_singleton([0])
    # assert is_singleton((0,))
    # assert is_singleton({0})
