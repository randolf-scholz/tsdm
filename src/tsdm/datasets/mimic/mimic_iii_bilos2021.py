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

__all__ = ["MIMIC_III_Bilos2021"]


import os
import subprocess
from getpass import getpass

import pandas as pd
from pandas import DataFrame

from tsdm.datasets.base import SingleTableDataset


class MIMIC_III_Bilos2021(SingleTableDataset):
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

    SOURCE_URL = r"https://physionet.org/content/mimiciii/get-zip/1.4/"
    INFO_URL = r"https://physionet.org/content/mimiciii/1.4/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"

    rawdata_files = ["complete_tensor.csv"]
    rawdata_hashes = {
        "complete_tensor.csv": "sha256:f2b09be20b021a681783d92a0091a49dcd23d8128011cb25990a61b1c2c1210f"
    }
    rawdata_schemas = {
        "complete_tensor.csv": {
            "UNIQUE_ID"  : "int16",
            "TIME_STAMP" : "int16",
            "LABEL_CODE" : "int16",
            "VALUENORM"  : "float32",
            "MEAN"       : "float32",
            "STD"        : "float32",
        }
    }  # fmt: skip

    rawdata_shapes = {"complete_tensor.csv": (3082224, 7)}
    table_names = ["timeseries"]
    table_hashes = {
        "timeseries": "pandas:-5464950709022187442",
    }
    table_shapes = {
        "timeseries": (552327, 96),
        "static_covariates": (96, 3),
    }

    def clean_table(self) -> DataFrame:
        self.LOGGER.info("Loading main file.")
        ts = pd.read_csv(self.rawdata_paths["complete_tensor.csv"], index_col=0)

        # Check shape.
        if ts.shape != self.rawdata_shapes["complete_tensor.csv"]:
            raise ValueError(
                f"The {ts.shape=} is not correct.Please apply the modified"
                " preprocessing using bin_k=2, as outlined inthe appendix. The"
                " resulting tensor should have 3082224 rows and 7 columns."
            )

        # Extract Original Data Table.
        ts = (
            ts.astype(self.rawdata_schemas["complete_tensor.csv"])
            .loc[:, ["UNIQUE_ID", "TIME_STAMP", "LABEL_CODE", "VALUENUM"]]
            .reset_index(drop=True)
            .set_index(["UNIQUE_ID", "TIME_STAMP"])
            .pivot(columns="LABEL_CODE", values="VALUENUM")
            .astype("float32")
            .sort_index()
            .sort_index(axis=1)
        )
        ts.columns = ts.columns.astype("string")

        # NOTE: For the MIMIC-III and MIMIC-IV datasets, Bilos et al. perform standardization
        #  over the full data slice, including test!
        # https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/lib/get_data.py
        ts = (ts - ts.mean()) / ts.std()

        return ts.astype("float32")

    def download_file(self, fname: str, /) -> None:
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
        user = input("MIMIC-III username: ")
        password = getpass(prompt="MIMIC-III password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N --cut-dirs"
            f" {cut_dirs} -P {self.RAWDATA_DIR!r} {self.SOURCE_URL} -O {path}",
            shell=True,
            check=True,
        )

        file = self.RAWDATA_DIR / "index.html"
        file.rename(fname)
