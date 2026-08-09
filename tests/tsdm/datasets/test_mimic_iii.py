import polars as pl

from tsdm.datasets import MIMIC_III_DeBrouwer2019


def test_mimic_iii_preprocessing() -> None:
    MIMIC_III_DeBrouwer2019.reset_dataset_files(force=True)
    ds = MIMIC_III_DeBrouwer2019()

    for key in MIMIC_III_DeBrouwer2019.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == MIMIC_III_DeBrouwer2019.table_schemas[key]
