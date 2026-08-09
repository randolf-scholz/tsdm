import polars as pl

from tsdm.datasets import ETT


def test_ett_preprocessing() -> None:
    ETT.reset_dataset_files(force=True)
    ds = ETT()

    for key in ETT.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ETT.table_schemas[key]
