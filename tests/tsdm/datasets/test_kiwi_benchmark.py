import polars as pl

from tsdm.datasets import KiwiBenchmark


def test_kiwi_benchmark_preprocessing() -> None:
    KiwiBenchmark.reset_dataset_files(force=True)
    ds = KiwiBenchmark()

    assert ds.timeseries.width == ds.timeseries_metadata.height
    assert ds.static_covariates.width == ds.static_covariates_metadata.height

    for key in ds.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
