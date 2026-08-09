r"""Test PhysioNet 2019."""

import polars as pl
import pytest

from tsdm.datasets import PhysioNet2019


@pytest.mark.manual
def test_physionet_2019_preprocessing() -> None:
    PhysioNet2019.reset_dataset_files(force=True)
    ds = PhysioNet2019()

    for key in PhysioNet2019.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == PhysioNet2019.table_schemas[key]

    assert ds.timeseries_metadata.height == ds.timeseries.width
    assert ds.static_covariates_metadata.height == ds.static_covariates.width
