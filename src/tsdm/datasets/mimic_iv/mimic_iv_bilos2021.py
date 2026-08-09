r"""MIMIC-IV clinical dataset.

Abstract
--------
Retrospectively collected medical data has the opportunity to improve patient care through knowledge discovery and
algorithm development. Broad reuse of medical data is desirable for the greatest public good, but data sharing must
be done in a manner which protects patient privacy. The Medical Information Mart for Intensive Care (MIMIC)-III
database provided critical care data for over 40,000 patients admitted to intensive care units at the
Beth Israel Deaconess Medical Center (BIDMC). Importantly, MIMIC-III was deidentified, and patient identifiers
were removed according to the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision.
MIMIC-III has been integral in driving large amounts of research in clinical informatics, epidemiology,
and machine learning. Here we present MIMIC-IV, an update to MIMIC-III, which incorporates contemporary data
and improves on numerous aspects of MIMIC-III. MIMIC-IV adopts a modular approach to data organization,
highlighting data provenance and facilitating both individual and combined use of disparate data sources.
MIMIC-IV is intended to carry on the success of MIMIC-III and support a broad set of applications within healthcare.
"""

__all__ = ["RAWDATA_SCHEMA", "TARGET_SCHEMA", "MIMIC_IV_Bilos2021"]


import os
import subprocess
from getpass import getpass
from itertools import product
from typing import Literal

import polars as pl

from tsdm.datasets.base import DatasetBase

RAWDATA_SCHEMA = {
    "hadm_id": pl.Float64,
    "time_stamp": pl.Int16,
    **{
        f"{kind}_label_{label}": pl.Float32
        for label, kind in product(range(102), ("Value", "Mask"))
    },
}
TARGET_SCHEMA = {
    "hadm_id": pl.Int32,
    "time_stamp": pl.Int16,
    # The 5σ outlier filter removes every observation from labels 37 and 71.
    **{f"Value_{label}": pl.Float32 for label in range(102) if label not in (37, 71)},
}


class MIMIC_IV_Bilos2021(DatasetBase[Literal["timeseries"], pl.DataFrame]):
    r"""MIMIC-IV Clinical Database.

    Retrospectively collected medical data has the opportunity to improve patient care through knowledge discovery and
    algorithm development. Broad reuse of medical data is desirable for the greatest public good, but data sharing must
    be done in a manner which protects patient privacy. The Medical Information Mart for Intensive Care (MIMIC)-III
    database provided critical care data for over 40,000 patients admitted to intensive care units at the
    Beth Israel Deaconess Medical Center (BIDMC). Importantly, MIMIC-III was deidentified, and patient identifiers
    were removed according to the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision.
    MIMIC-III has been integral in driving large amounts of research in clinical informatics, epidemiology,
    and machine learning. Here we present MIMIC-IV, an update to MIMIC-III, which incorporates contemporary data
    and improves on numerous aspects of MIMIC-III. MIMIC-IV adopts a modular approach to data organization,
    highlighting data provenance and facilitating both individual and combined use of disparate data sources.
    MIMIC-IV is intended to carry on the success of MIMIC-III and support a broad set of applications within healthcare.
    """

    SOURCE_URL = r"https://physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL = r"https://physionet.org/content/mimiciv/1.0/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"

    table_names = ["timeseries"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["full_dataset.csv"]
    rawdata_hashes = {
        "full_dataset.csv": "sha256:f2b09be20b021a681783d92a0091a49dcd23d8128011cb25990a61b1c2c1210f"
    }
    rawdata_schemas = {"full_dataset.csv": RAWDATA_SCHEMA}
    rawdata_shapes = {"full_dataset.csv": (2_485_649, 206)}
    table_schemas = {"timeseries": TARGET_SCHEMA}
    table_shapes = {"timeseries": (2_485_649, 102)}

    def clean_timeseries(self) -> pl.DataFrame:
        self.LOGGER.info("Loading main file.")
        fname = "full_dataset.csv"
        rawdata_schema = self.rawdata_schemas[fname]
        rawdata_shape = self.rawdata_shapes[fname]
        table = pl.read_csv(
            self.rawdata_paths[fname],
            schema=rawdata_schema,
        ).fill_nan(None)

        if table.shape != rawdata_shape:
            raise ValueError(f"{table.shape=} does not match {rawdata_shape=}.")

        value_columns = [f"Value_label_{label}" for label in range(102)]
        mask_columns = [f"Mask_label_{label}" for label in range(102)]
        target_columns = [f"Value_{label}" for label in range(102)]

        if missing_values := set(value_columns) - set(table.columns):
            raise ValueError(f"Value columns not found: {missing_values}")

        if missing_masks := set(mask_columns) - set(table.columns):
            raise ValueError(f"Mask columns not found: {missing_masks}")

        # fold masks into value column and rename.
        masked = (
            table.lazy()
            .select(
                pl.col("hadm_id").cast(pl.Int32).alias("hadm_id"),
                pl.col("time_stamp"),
                *(
                    pl.when(pl.col(f"Mask_label_{k}").eq(1))
                    .then(pl.col(f"Value_label_{k}"))
                    .otherwise(None)
                    .alias(f"Value_{k}")
                    for k in range(102)
                ),
            )
            .filter(
                pl.any_horizontal(
                    pl.col(column).is_not_null() for column in target_columns
                )
            )
        )

        # NOTE: For the MIMIC-III and MIMIC-IV datasets, Bilos et al. perform standardization
        #  over the full data slice, including test!
        # https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/lib/get_data.py
        normalized = masked.select(
            "hadm_id",
            "time_stamp",
            *(
                ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std(ddof=1))
                for column in target_columns
            ),
        )

        # NOTE: For the MIMIC-IV dataset, Bilos et al. drop 5σ-outliers.
        table = normalized.select(
            "hadm_id",
            "time_stamp",
            *(
                pl.when(pl.col(column).is_between(-5, 5, closed="none"))
                .then(pl.col(column))
                .otherwise(None)
                for column in target_columns
            ),
        ).collect()

        return table.select(*TARGET_SCHEMA).sort("hadm_id", "time_stamp")

    def load_table(self, key: Literal["timeseries"], /) -> pl.DataFrame:
        r"""Load a cleaned table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])

    def get_rawdata_file(self, fname: str, /) -> None:
        if not self.rawdata_files_exist():
            raise RuntimeError(
                "Please manually apply the preprocessing code found at"
                f" {self.GITHUB_URL}.\nPut the resulting file 'complete_tensor.csv' in"
                f" {self.RAWDATA_DIR}.\nThe cleaning code is not included in this"
                " package because the original.\nauthors did not provide a license"
                " for it."
            )

        path = self.rawdata_paths[fname]
        cut_dirs = self.SOURCE_URL.count("/") - 3
        user = input("MIMIC-IV username: ")
        password = getpass(prompt="MIMIC-IV password: ", stream=None)
        os.environ["PASSWORD"] = password
        subprocess.run(
            [
                "/usr/bin/wget",
                "--user", user,
                "--password", "$PASSWORD",
                "--cut-dirs", str(cut_dirs),  # ignore the first 3 directories
                "-P", str(self.RAWDATA_DIR),  # directory prefix
                "-O", str(path),  # output document (zip file)
                "-c",   # continue
                "-r",   # recursive
                "-np",  # don't ascend to the parent directory
                "-nH",  # don't create host directories
                "-N",   # don't re-retrieve files unless newer than local
                self.SOURCE_URL,
            ],
            check=True,
        )  # fmt: skip

        file = self.RAWDATA_DIR / "index.html"
        file.rename(fname)
