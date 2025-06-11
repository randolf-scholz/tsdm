r"""Test dataset protocols."""

from typing import Literal, assert_type

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch as pt

from tsdm.data import Indexable, MapDataset, PandasDataset


def static_test_indexabledataset() -> None:
    r"""Test object."""
    list_int = [1, 2, 3]
    list_str = ["a", "b", "c"]
    tuple_int = (1, 2, 3)
    tuple_str = ("a", "b", "c")
    pt_tensor = pt.tensor([1, 2, 3])
    np_array = np.array([1, 2, 3])
    pd_series = pd.Series([1, 2, 3], index=[-4, 2, -17])
    pa_array = pa.array([1, 2, 3])
    pl_series = pl.Series("dummy", ["a", "b", "c"])

    def as_idxdataset[T](x: Indexable[T], /) -> Indexable[T]:
        return x

    assert_type(as_idxdataset(list_int), Indexable[int])
    assert_type(as_idxdataset(list_str), Indexable[str])
    assert_type(as_idxdataset(tuple_int), Indexable[int])  # pyright: ignore[reportAssertTypeFailure]
    assert_type(as_idxdataset(tuple_str), Indexable[str])  # pyright: ignore[reportAssertTypeFailure]
    assert_type(as_idxdataset(tuple_int), Indexable[Literal[1, 2, 3]])  # type: ignore[assert-type]
    assert_type(as_idxdataset(tuple_str), Indexable[Literal["a", "b", "c"]])  # type: ignore[assert-type]
    assert_type(as_idxdataset(pt_tensor), Indexable[pt.Tensor])  # type: ignore[assert-type]
    assert_type(as_idxdataset(np_array), Indexable)
    assert_type(as_idxdataset(pd_series), Indexable)
    assert_type(as_idxdataset(pa_array), Indexable)
    assert_type(as_idxdataset(pl_series), Indexable)


def static_test_mapdataset() -> None:
    r"""Test object."""
    dict_int_str = {1: "a", 2: "b", 3: "c"}
    dict_str_int = {"a": 1, "b": 2, "c": 3}
    pd_series = pd.Series([1, 2, 3], index=["foo", "bar", "baz"])

    def as_mapdataset[K, V](x: MapDataset[K, V], /) -> MapDataset[K, V]:
        return x

    assert_type(as_mapdataset(dict_int_str), MapDataset[int, str])
    assert_type(as_mapdataset(dict_str_int), MapDataset[str, int])
    assert_type(as_mapdataset(pd_series), MapDataset)


def static_test_pandasdataset() -> None:
    pd_series = pd.Series(["a", "b", "c"])
    pd_frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def as_pandasdataset[K, V](x: PandasDataset[K, V], /) -> PandasDataset[K, V]:
        return x

    assert_type(as_pandasdataset(pd_series), PandasDataset)
    assert_type(as_pandasdataset(pd_frame), PandasDataset)
