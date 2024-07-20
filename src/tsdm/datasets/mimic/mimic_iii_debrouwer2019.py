r"""MIMIC-III Clinical Database.

Abstract
--------
MIMIC-III is a large, freely available database comprising de-identified health-related
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

__all__ = ["MIMIC_III_DeBrouwer2019"]


import os
import subprocess
from getpass import getpass
from typing import Literal

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pandas import DataFrame

from tsdm.datasets.base import MultiTableDataset

type Key = Literal["timeseries", "metadata"]


class MIMIC_III_DeBrouwer2019(MultiTableDataset[Key, DataFrame]):
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

    Notes:
        NOTE: ``TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))``
        and ``bin_k = 10``
        i.e. ``TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)``
        i.e. ``TIME_STAMP ≈ 10*total_hours``
        so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    SOURCE_URL = r"https://physionet.org/content/mimiciii/get-zip/1.4/"
    INFO_URL = r"https://physionet.org/content/mimiciii/1.4/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/edebrouwer/gru_ode_bayes/"

    table_names = ["timeseries", "metadata"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["complete_tensor.csv"]
    rawdata_hashes = {
        "complete_tensor.csv": "sha256:8e884a916d28fd546b898b54e20055d4ad18d9a7abe262e15137080e9feb4fc2",
    }
    rawdata_shapes = {"complete_tensor.csv": (3082224, 7)}
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
    dataset_hashes = {  # pyright: ignore[reportAssignmentType]
        "timeseries": "sha256:2ebb7da820560f420f71c0b6fb068a46449ef89b238e97ba81659220fae8151b",
        "metadata": "sha256:4779aa3639f468126ea263645510d5395d85b73caf1c7abb0a486561b761f5b4",
    }
    table_shapes = {  # pyright: ignore[reportAssignmentType]
        "timeseries": (552327, 96),
        "metadata": (96, 3),
    }

    KEYS = ["timeseries", "metadata"]

    timeseries: DataFrame
    metadata: DataFrame

    def clean_table(self, key: Key) -> DataFrame:
        if key == "metadata":
            return self.timeseries.describe().T.astype("float32")

        if key != "timeseries":
            raise KeyError(f"{key=} is not a valid key.")

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

    # FIXME: https://github.com/numpy/numpy/issues/24738
    def make_histograms(self) -> tuple[Figure, NDArray]:
        r"""Make histograms of the timeseries."""
        fig, axes = plt.subplots(
            16,
            6,
            figsize=(20, 32),
            constrained_layout=True,
            sharey=True,
            squeeze=False,
        )

        for col, ax in zip(self.timeseries, axes.flatten(), strict=True):
            self.timeseries[col].hist(ax=ax, density=True, log=True, bins=20)
            ax.set_ylim(10**-6, 1)

        return fig, axes
