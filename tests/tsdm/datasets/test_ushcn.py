r"""Test tsdm.datasets.USHCN."""

import polars as pl
import pytest

from tsdm.datasets import USHCN, USHCN_DeBrouwer2019


def test_ushcn_debrouwer2019_preprocessing() -> None:
    USHCN_DeBrouwer2019.reset_dataset_files(force=True)
    ds = USHCN_DeBrouwer2019()

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]


@pytest.mark.manual
def test_ushcn_preprocessing() -> None:
    USHCN.reset_dataset_files(force=True)
    ds = USHCN()

    assert ds.timeseries_metadata.height == ds.timeseries.width

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]


@pytest.mark.slow
def test_ushcn_full() -> None:
    dataset = USHCN()
    assert isinstance(dataset, USHCN)
