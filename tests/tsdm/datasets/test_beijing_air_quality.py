import polars as pl

from tsdm.datasets import BeijingAirQuality


def test_beijing_air_quality():
    BeijingAirQuality.reset_dataset_files(force=True)
    ds = BeijingAirQuality()

    assert isinstance(ds.timeseries, pl.DataFrame)
    assert isinstance(ds.timeseries_metadata, pl.DataFrame)
    assert dict(ds.timeseries.schema) == BeijingAirQuality.table_schemas["timeseries"]
    assert (
        dict(ds.timeseries_metadata.schema)
        == (BeijingAirQuality.table_schemas["timeseries_metadata"])
    )
