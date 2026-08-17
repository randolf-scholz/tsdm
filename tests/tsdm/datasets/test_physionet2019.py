r"""Test PhysioNet 2019."""

import polars as pl
import pytest

from tsdm.datasets import PhysioNet2019


@pytest.mark.manual
def test_physionet_2019_preprocessing() -> None:
    PhysioNet2019.reset_dataset_files(force=True)
    ds = PhysioNet2019()

    assert ds.timeseries_metadata.height == ds.timeseries.width
    assert ds.static_covariates_metadata.height == ds.static_covariates.width

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
