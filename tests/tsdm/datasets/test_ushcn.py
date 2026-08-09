r"""Test tsdm.datasets.USHCN."""

import polars as pl
import pytest

from tsdm.datasets import USHCN, USHCN_DeBrouwer2019


def test_ushcn_debrouwer2019_preprocessing() -> None:
    USHCN_DeBrouwer2019.reset_dataset_files(force=True)
    ds = USHCN_DeBrouwer2019()

    for key in USHCN_DeBrouwer2019.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == USHCN_DeBrouwer2019.table_schemas[key]


def test_ushcn_preprocessing() -> None:
    USHCN.reset_dataset_files(force=True)
    ds = USHCN()

    for key in USHCN.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == USHCN.table_schemas[key]


@pytest.mark.slow
def test_ushcn_full() -> None:
    dataset = USHCN()
    assert isinstance(dataset, USHCN)
