import polars as pl

from tsdm.datasets import BeijingAirQuality


def test_beijing_air_quality_preprocessing() -> None:
    BeijingAirQuality.reset_dataset_files(force=True)
    ds = BeijingAirQuality()

    for key in BeijingAirQuality.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == BeijingAirQuality.table_schemas[key]

    assert ds.timeseries_metadata.height == ds.timeseries.width
