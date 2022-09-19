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


import warnings
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from tsdm.datasets.base import MultiFrameDataset


class MIMIC_IV_Bilos2021(MultiFrameDataset):
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

    BASE_URL = r"https://www.physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL = r"https://www.physionet.org/content/mimiciv/1.0/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"
    SHA256 = "cb90e0cef16d50011aaff7059e73d3f815657e10653a882f64f99003e64c70f5"
    SHAPE = (2485769, 206)

    dataset_files = {"timeseries": "timeseries.parquet"}
    rawdata_files = r"full_dataset.csv"
    rawdata_paths: Path
    index = ["timeseries"]

    def _clean(self, key):
        if not self.rawdata_paths.exists():
            raise RuntimeError(
                f"Please apply the preprocessing code found at {self.GITHUB_URL}."
                f"\nPut the resulting file 'complete_tensor.csv' in {self.RAWDATA_DIR}."
            )

        if sha256(self.rawdata_paths.read_bytes()).hexdigest() != self.SHA256:
            warnings.warn("The sha256 seems incorrect.")

        ts = pd.read_csv(self.rawdata_paths)

        if ts.shape != self.SHAPE:
            raise ValueError(f"The {ts.shape=} is not correct.")

        ts = ts.sort_values(by=["hadm_id", "time_stamp"])
        ts = ts.astype(
            {
                "hadm_id": "int32",
                "time_stamp": "int16",
            }
        )
        ts = ts.set_index(list(ts.columns[:2]))
        for i, col in enumerate(ts):
            if i % 2 == 1:
                continue
            ts[col] = np.where(ts.iloc[:, i + 1], ts[col], np.nan)
        ts = ts.drop(columns=ts.columns[1::2])
        ts = ts.sort_index()
        ts = ts.astype("float32")
        ts.to_parquet(self.dataset_paths["timeseries"])

    def _load(self, key):
        return pd.read_parquet(self.dataset_paths[key])

    def _download(self, **kwargs):
        if not self.rawdata_paths.exists():
            raise RuntimeError(
                f"Please apply the preprocessing code found at {self.GITHUB_URL}."
                f"\nPut the resulting file 'complete_tensor.csv' in {self.RAWDATA_DIR}."
            )
