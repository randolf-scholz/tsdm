r"""MIMIC-III Clinical Database.

Abstract
--------

MIMIC-III is a large, freely-available database comprising de-identified health-related
data associated with over forty thousand patients who stayed in critical care units of
the Beth Israel Deaconess Medical Center between 2001 and 2012.
The database includes information such as demographics, vital sign measurements made at
the bedside (~1 data point per hour), laboratory test results, procedures, medications,
caregiver notes, imaging reports, and mortality (including post-hospital discharge).

MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical
decision-rule improvement, and electronic tool development. It is notable for three
factors: it is freely available to researchers worldwide; it encompasses a diverse and
very large population of ICU patients; and it contains highly granular data, including
vital signs, laboratory results, and medications.
"""

__all__ = ["MIMIC_III"]

import os
import subprocess
from getpass import getpass

import pandas as pd

from tsdm.datasets.base import Dataset


class MIMIC_III(Dataset):
    """MIMIC-III Clinical Database.

    MIMIC-III is a large, freely-available database comprising deidentified health-related data
    associated with over forty thousand patients who stayed in critical care units of the Beth
    Israel Deaconess Medical Center between 2001 and 2012. The database includes information such
    as demographics, vital sign measurements made at the bedside (~1 data point per hour),
    laboratory test results, procedures, medications, caregiver notes, imaging reports, and
    mortality (including post-hospital discharge).

    MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical decision-rule
    improvement, and electronic tool development. It is notable for three factors: it is freely
    available to researchers worldwide; it encompasses a diverse and very large population of ICU
    patients; and it contains highly granular data, including vital signs, laboratory results,
    and medications.
    """

    base_url: str = r"https://physionet.org/content/mimiciii/get-zip/1.4/"
    info_url: str = r"https://physionet.org/content/mimiciii/1.4/"
    dataset_files = {"observations": "observations.feather", "stats": "stats.feather"}
    rawdata_files = "mimic-iii-clinical-database-1.4.zip"
    index = ["observations", "stats"]

    def _clean(self, key):
        ts_file = self.rawdata_dir / "complete_tensor.csv"
        if not ts_file.exists():
            raise RuntimeError(
                "Please apply the preprocessing code found at "
                "https://github.com/edebrouwer/gru_ode_bayes/."
                f"\nPut the resulting file 'complete_tensor.csv' in {self.rawdata_dir}."
            )

        ts = pd.read_csv(ts_file)
        ts = ts.sort_values(by=["UNIQUE_ID", "TIME_STAMP"])
        ts = ts.astype(
            {
                "UNIQUE_ID": "int16",
                "TIME_STAMP": "int16",
                "LABEL_CODE": "int16",
                "VALUENORM": "float32",
                "MEAN": "float32",
                "STD": "float32",
            }
        )

        means = ts.groupby("LABEL_CODE").mean()["VALUENUM"].rename("MEANS")
        stdvs = ts.groupby("LABEL_CODE").std()["VALUENUM"].rename("STDVS")
        stats = pd.DataFrame([means, stdvs]).T.reset_index()
        stats = stats.astype(
            {
                "LABEL_CODE": "int16",
                "MEANS": "float32",
                "STDVS": "float32",
            }
        )

        ts = ts[["UNIQUE_ID", "TIME_STAMP", "LABEL_CODE", "VALUENORM"]]
        ts = ts.reset_index(drop=True)
        stats.to_feather(self.dataset_paths["stats"])
        ts.to_feather(self.dataset_paths["observations"])

    def _load(self, key):
        # return NotImplemented
        return pd.read_feather(self.dataset_paths[key])

    def _download(self, **kwargs):
        cut_dirs = self.base_url.count("/") - 3
        user = input("MIMIC-III username: ")
        password = getpass(prompt="MIMIC-III password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            + f"--cut-dirs {cut_dirs} -P '{self.rawdata_dir}' {self.base_url} ",
            shell=True,
            check=True,
        )

        file = self.rawdata_dir / "index.html"
        os.rename(file, self.rawdata_files)
