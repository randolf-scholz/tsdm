import polars as pl

from tsdm.datasets import Traffic


def test_traffic_preprocessing() -> None:
    Traffic.reset_dataset_files(force=True)
    ds = Traffic()

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
