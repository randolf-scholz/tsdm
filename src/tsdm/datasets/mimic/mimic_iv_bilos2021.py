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

__all__ = ["MIMIC_IV_Bilos2021"]


import os
import subprocess
from getpass import getpass

import numpy as np
from pandas import DataFrame
from pyarrow import Table, csv

from tsdm.datasets.base import SingleTableDataset


class MIMIC_IV_Bilos2021(SingleTableDataset):
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

    BASE_URL = r"https://physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL = r"https://physionet.org/content/mimiciv/1.0/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"

    rawdata_files = ["full_dataset.csv"]
    rawdata_hashes = {
        "full_dataset.csv": (
            "sha256:f2b09be20b021a681783d92a0091a49dcd23d8128011cb25990a61b1c2c1210f"
        )
    }
    rawdata_schemas = {
        "full_dataset.csv": {
            "hadm_id": "int32[pyarrow]",
            "time_stamp": "int16[pyarrow]",
        }
    }

    rawdata_shape = (2485649, 206)
    table_names = ["timeseries"]
    table_hashes = {
        "timeseries": "pandas:-5464950709022187442",
    }
    table_shapes = {
        "timeseries": (2485649, 102),
    }

    def clean_table(self) -> DataFrame:
        self.LOGGER.info("Loading main file.")
        table: Table = csv.read_csv(self.rawdata_paths["full_dataset.csv"])

        if table.shape != self.rawdata_shape:
            raise ValueError(f"{table.shape=} does not match {self.rawdata_shape=}.")

        # Convert to pandas.
        ts = (
            table.to_pandas(self_destruct=True)
            .astype(self.rawdata_schemas["full_dataset.csv"])
            .set_index(["hadm_id", "time_stamp"])
            .sort_index()
        )

        # Remove mask columns, replace values with nan.
        # Original labels: Value_label_k, Mask_label_k for k in 0, ..., 99.
        for i, col in enumerate(ts):
            if i % 2 == 1:
                continue
            assert ts.columns[i + 1] == col.replace("Value", "Mask")
            ts[col] = np.where(ts.iloc[:, i + 1], ts[col], np.nan)

        # Drop mask columns.
        ts = (
            ts.drop(columns=ts.columns[1::2])
            .dropna(how="all")
            .astype("float32")
            .sort_index(axis="columns")
        )

        # NOTE: For the MIMIC-III and MIMIC-IV datasets, Bilos et al. perform standardization
        #  over the full data slice, including test!
        # https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/lib/get_data.py
        ts = (ts - ts.mean()) / ts.std()

        # NOTE: For the MIMIC-IV dataset, Bilos et al. drop 5σ-outliers.
        ts = ts[(-5 < ts) & (ts < 5)].dropna(axis=1, how="all").copy()

        # NOTE: only numpy float types supported by torch
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

        cut_dirs = self.BASE_URL.count("/") - 3
        user = input("MIMIC-IV username: ")
        password = getpass(prompt="MIMIC-IV password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            + f"--cut-dirs {cut_dirs} -P {self.RAWDATA_DIR!r} {self.BASE_URL} -O"
            f" {path}",
            shell=True,
            check=True,
        )

        file = self.RAWDATA_DIR / "index.html"
        os.rename(file, fname)


# class MIMIC_IV_Bilos2021(MIMIC_IV):
#     RAWDATA_DIR = MIMIC_IV.RAWDATA_DIR
#     __version__: Final[str] = "1.0"
#     rawdata_hashes = {
#         "mimic-iv-1.0.zip": (
#             "sha256:dd226e8694ad75149eed2840a813c24d5c82cac2218822bc35ef72e900baad3d"
#         ),
#     }
#
#     mimic_iv: MIMIC_IV
#
#     @cached_property
#     def rawdata(self):
#         self.mimic_iv = MIMIC_IV()
#         return self.mimic_iv.tables
#
#     def clean_admissions(self):
#         adm = self.rawdata["admissions"]
#         patients_df = self.rawdata["patients"]
#         chartevents = self.rawdata["chartevents"]
#
#         # keep only patients present in patients data
#         adm_dob = pd.merge(
#             patients_df[["subject_id", "anchor_age"]], adm, on="subject_id"
#         )
#
#         df = adm.groupby("subject_id")["hadm_id"].nunique()
#         subj_ids = list(df[df == 1].index)
#
#         adm_1 = adm_dob.assign(
#             elapsed_days=(adm_dob["dischtime"] - adm_dob["admittime"]).dt.days
#         ).loc[adm_dob["subject_id"].isin(subj_ids)]
#
#         adm_2 = adm_1.loc[
#             (adm_1["elapsed_days"] > 2)
#             & (adm_1["elapsed_days"] < 30)
#             # only patients older than 15
#             & (adm_1["anchor_age"] > 15)
#             & (adm_1["hadm_id"].isin(chartevents["hadm_id"]))
#         ]
#
#         return adm_2
