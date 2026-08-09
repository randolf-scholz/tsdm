import polars as pl

from tsdm.datasets import Electricity


def test_electricity():
    Electricity.reset_dataset_files(force=True)
    ds = Electricity()

    assert isinstance(ds.timeseries, pl.DataFrame)
    assert dict(ds.timeseries.schema) == Electricity.table_schemas["timeseries"]
