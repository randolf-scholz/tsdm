import polars as pl

from tsdm.datasets import MIMIC_IV_Bilos2021


def test_mimic_iv_preprocessing() -> None:
    MIMIC_IV_Bilos2021.reset_dataset_files(force=True)
    ds = MIMIC_IV_Bilos2021()

    for key in MIMIC_IV_Bilos2021.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        # assert dict(ds[key].schema) == MIMIC_IV_Bilos2021.table_schemas[key]
