"""Check pandas MultiIndex Slicing."""

import numpy as np
import pandas as pd
import pytest


def test_multiindex_slicing_0level() -> None:
    """Slice a MultiIndex with a Boolean Series with Index => fails!."""
    index = pd.MultiIndex.from_product(
        [["A", "B", "C"], ["x", "y"]], names=["outer", "inner"]
    )
    data = np.random.randn(len(index), 3)
    df = pd.DataFrame(data, index, columns=["foo", "bar", "baz"])
    mask_index = pd.Index(["A", "B", "C"], name="outer")
    mask = pd.Series([True, False, True], index=mask_index)
    with pytest.raises(pd.errors.IndexingError):
        _ = df.loc[mask]


def test_multiindex_slicing_1level():
    """Slice a MultiIndex with a Boolean Series with MultiIndex with 1 level => works!."""
    index = pd.MultiIndex.from_product(
        [["A", "B", "C"], ["x", "y"]], names=["outer", "inner"]
    )
    data = np.random.randn(len(index), 3)
    df = pd.DataFrame(data, index, columns=["foo", "bar", "baz"])
    mask_index = pd.MultiIndex.from_product([["A", "B", "C"]], names=["outer"])
    mask = pd.Series([True, False, True], index=mask_index)
    _ = df.loc[mask]


def test_multiindex_slicing_2levels():
    """Slice a MultiIndex with a Boolean Series with MultiIndex with multiple levels => works!."""
    index = pd.MultiIndex.from_product(
        [["A", "B", "C"], ["x", "y"], [0, 1, 2]], names=["outer", "inner", "step"]
    )
    data = np.random.randn(len(index), 3)
    df = pd.DataFrame(data, index, columns=["foo", "bar", "baz"])
    mask_index = pd.MultiIndex.from_product(
        [["A", "B", "C"], ["x", "y"]], names=["outer", "inner"]
    )
    mask = pd.Series(np.random.randn(len(mask_index)) > 0.5, index=mask_index)
    _ = df.loc[mask]
