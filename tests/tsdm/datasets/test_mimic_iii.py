import polars as pl
import pytest

from tsdm.datasets import MIMIC_III


@pytest.mark.manual
def test_mimic_iii_preprocessing() -> None:
    MIMIC_III.reset_dataset_files(force=True)
    ds = MIMIC_III()

    for key in ds.table_names:
        assert isinstance(ds[key], pl.LazyFrame)
        assert dict(ds[key].schema) == ds.table_schemas[key]
