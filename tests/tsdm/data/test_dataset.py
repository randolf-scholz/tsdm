r"""Test dataset protocols."""

import logging
from collections.abc import Mapping

import pandas as pd
from pytest import mark

from tsdm.data import MapDataset, PandasDataset
from tsdm.types.protocols import MappingProtocol, assert_protocol

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


DATASETS = {
    "series": pd.Series(["a", "b", "c"], index=[-4, 2, -17]),
    "dataframe": pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], index=["a", "b"], columns=["x", "y", "z"]
    ),
}


@mark.parametrize("name", DATASETS)
def test_map_dataset(name: str) -> None:
    r"""Test object."""
    dataset = DATASETS[name]
    assert_protocol(dataset, MapDataset)


@mark.parametrize("name", DATASETS)
def test_pandas_dataset(name: str) -> None:
    r"""Test object."""
    dataset = DATASETS[name]
    assert_protocol(dataset, PandasDataset)


@mark.parametrize("name", DATASETS)
def test_pandas_mapping(name: str) -> None:
    r"""Test object."""
    dataset = DATASETS[name]
    assert_protocol(dataset, MappingProtocol)


def test_map_dataset_mapping():
    data: Mapping[str, int] = {"a": 1, "b": 2, "c": 3}
    assert isinstance(data, Mapping)
    assert isinstance(data, MapDataset)

    dataset: MapDataset[str, int] = data
    assert isinstance(dataset, MapDataset)

    class BareMapDataset:
        """Bare implementation of MapDataset."""

        data = {"a": 1, "b": 2, "c": 3}

        def __len__(self):
            return len(self.data)

        def keys(self):
            return self.data.keys()

        def __getitem__(self, key):
            return self.data[key]

    dataset2 = BareMapDataset()
    assert isinstance(dataset2, MapDataset)
