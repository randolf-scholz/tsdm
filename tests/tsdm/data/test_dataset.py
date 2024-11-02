r"""Test dataset protocols."""

import logging
from collections.abc import KeysView, Mapping
from typing import assert_type

import pandas as pd
import pytest

from tsdm.data import Dataset, IndexableDataset, MapDataset, PandasDataset
from tsdm.testing import assert_protocol
from tsdm.types.protocols import Map

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


DATASETS = {
    "series": pd.Series(["a", "b", "c"], index=[-4, 2, -17]),
    "dataframe": pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], index=["a", "b"], columns=["x", "y", "z"]
    ),
}


def static_test_dataset() -> None:
    r"""Test object."""
    dict_int_str = {1: "a", 2: "b", 3: "c"}
    dict_str_int = {"a": 1, "b": 2, "c": 3}
    list_int: IndexableDataset[int] = [1, 2, 3]
    list_str = ["a", "b", "c"]

    def as_idxdataset[T](x: IndexableDataset[T], /) -> IndexableDataset[T]:
        return x

    def as_mapdataset[K, V](x: MapDataset[K, V], /) -> MapDataset[K, V]:
        return x

    def as_dataset[T](x: Dataset[T], /) -> Dataset[T]:
        return x

    assert_type(as_idxdataset(list_int), IndexableDataset[int])
    assert_type(as_idxdataset(list_str), IndexableDataset[str])

    assert_type(as_mapdataset(dict_int_str), MapDataset[int, str])
    assert_type(as_mapdataset(dict_str_int), MapDataset[str, int])

    assert_type(as_dataset(dict_int_str), Dataset[str])
    assert_type(as_dataset(dict_str_int), Dataset[int])
    assert_type(as_dataset(list_int), Dataset[int])
    assert_type(as_dataset(list_str), Dataset[str])


@pytest.mark.parametrize("name", DATASETS)
def test_map_dataset(name: str) -> None:
    r"""Test object."""
    dataset = DATASETS[name]
    assert_protocol(dataset, MapDataset)


@pytest.mark.parametrize("name", DATASETS)
def test_pandas_dataset(name: str) -> None:
    r"""Test object."""
    dataset = DATASETS[name]
    assert_protocol(dataset, PandasDataset)


@pytest.mark.parametrize("name", DATASETS)
def test_pandas_mapping(name: str) -> None:
    r"""Test object."""
    dataset = DATASETS[name]
    assert_protocol(dataset, Map)


def test_map_dataset_mapping() -> None:
    data: Mapping[str, int] = {"a": 1, "b": 2, "c": 3}
    assert isinstance(data, Mapping)
    assert isinstance(data, MapDataset)

    dataset: MapDataset[str, int] = data
    assert isinstance(dataset, MapDataset)

    class BareMapDataset:
        def __init__(self) -> None:
            self.data = {"a": 1, "b": 2, "c": 3}

        def __len__(self) -> int:
            return len(self.data)

        def keys(self) -> KeysView[str]:
            return self.data.keys()

        def __getitem__(self, key: str) -> int:
            return self.data[key]

    dataset2 = BareMapDataset()
    assert isinstance(dataset2, MapDataset)
