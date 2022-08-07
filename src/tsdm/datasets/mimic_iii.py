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

from tsdm.datasets.base import MultiFrameDataset
from tsdm.encoders import TripletDecoder


class MIMIC_III(MultiFrameDataset):
    r"""MIMIC-III Clinical Database.

    MIMIC-III is a large, freely-available database comprising de-identified health-related data
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

    Notes
    -----
    NOTE: TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))
    and bin_k = 10
    i.e. TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)
    i.e. TIME_STAMP ≈ 10*total_hours
    so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    BASE_URL: str = r"https://physionet.org/content/mimiciii/get-zip/1.4/"
    INFO_URL: str = r"https://physionet.org/content/mimiciii/1.4/"
    dataset_files = {"observations": "observations.feather", "stats": "stats.feather"}
    rawdata_files = "mimic-iii-clinical-database-1.4.zip"
    index = ["observations", "stats"]

    def _clean(self, key):
        ts_file = self.RAWDATA_DIR / "complete_tensor.csv"
        if not ts_file.exists():
            raise RuntimeError(
                "Please apply the preprocessing code found at "
                "https://github.com/edebrouwer/gru_ode_bayes/."
                f"\nPut the resulting file 'complete_tensor.csv' in {self.RAWDATA_DIR}."
            )

        ts = pd.read_csv(ts_file, index_col=0)

        if ts.shape != (3082224, 7):
            raise ValueError(
                f"The {ts.shape=} is not correct."
                "Please apply the modified preprocessing using bin_k=2, as outlined in"
                "the appendix. The resulting tensor should have 3082224 rows and 7 columns."
            )

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
        ts = ts.set_index(["UNIQUE_ID", "TIME_STAMP"])
        ts = ts.sort_index()
        encoder = TripletDecoder(value_name="VALUENORM", var_name="LABEL_CODE")
        encoder.fit(ts)
        encoded = encoder.encode(ts)
        ts = encoded.reset_index()
        ts.columns = ts.columns.astype("string")
        stats.to_feather(self.dataset_paths["stats"])
        ts.to_feather(self.dataset_paths["observations"])

    def _load(self, key):
        # return NotImplemented
        df = pd.read_feather(self.dataset_paths[key])

        if key == "observations":
            df = df.set_index(["UNIQUE_ID", "TIME_STAMP"])
            df = df.sort_index()
        elif key == "stats":
            df = df.set_index("LABEL_CODE")
        return df

    def _download(self, **_):
        cut_dirs = self.BASE_URL.count("/") - 3
        user = input("MIMIC-III username: ")
        password = getpass(prompt="MIMIC-III password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            + f"--cut-dirs {cut_dirs} -P '{self.RAWDATA_DIR}' {self.BASE_URL} ",
            shell=True,
            check=True,
        )

        file = self.RAWDATA_DIR / "index.html"
        os.rename(file, self.rawdata_files)
