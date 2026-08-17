import polars as pl

from tsdm.datasets import Electricity


def test_electricity_preprocessing() -> None:
    Electricity.reset_dataset_files(force=True)
    ds = Electricity()

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
