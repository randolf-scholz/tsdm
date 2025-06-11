r"""Test dataset protocols."""

import logging
from collections.abc import KeysView, Mapping

import pandas as pd
import pytest

from tsdm.data import MapDataset, PandasDataset
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
