import polars as pl

from tsdm.datasets import KiwiBenchmark


def test_insilico_preprocessing() -> None:
    KiwiBenchmark.reset_dataset_files(force=True)
    ds = KiwiBenchmark()

    assert ds.timeseries_metadata.height == ds.timeseries.width

    for key in KiwiBenchmark.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == KiwiBenchmark.table_schemas[key]
