r"""Test tsdm.datasets.USHCN."""

import polars as pl
import pytest

from tsdm.datasets import USHCN


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
