import polars as pl

from tsdm.datasets import BeijingAirQuality


def test_beijing_air_quality_preprocessing() -> None:
    BeijingAirQuality.reset_dataset_files(force=True)
    ds = BeijingAirQuality()

    assert ds.timeseries_metadata.height == ds.timeseries.width

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
