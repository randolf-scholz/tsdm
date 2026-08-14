import polars as pl

from tsdm.datasets import MIMIC_III


def test_mimic_iii_preprocessing() -> None:
    MIMIC_III.reset_dataset_files(force=True)
    ds = MIMIC_III()

    for key in MIMIC_III.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == MIMIC_III.table_schemas[key]
