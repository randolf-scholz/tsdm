r"""Test PhysioNet 2012."""

import polars as pl
import pytest

from tsdm.datasets import PhysioNet2012


@pytest.mark.manual
def test_physionet_2012_preprocessing() -> None:
    PhysioNet2012.reset_dataset_files(force=True)
    ds = PhysioNet2012()

    assert ds.timeseries_metadata.height == ds.timeseries.width
    assert ds.static_covariates_metadata.height == ds.static_covariates.width

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
