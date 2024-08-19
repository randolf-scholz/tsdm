r"""Custom processed version of the MIMIC-III dataset."""

__all__ = [
    # Classes
    "MIMIC_III_Scholz2024",
]

from typing import Any

import numpy as np
import pandas as pd

from tsdm.datasets.mimic.mimic_iii import MIMIC_III
from tsdm.types.dtypes import map_pandas_arrowtime_numpy


class MIMIC_III_Scholz2024(MIMIC_III):
    r"""Custom processed version of the MIMIC-III dataset."""

    RAWDATA_DIR = MIMIC_III.RAWDATA_DIR

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the dataset."""
        super().__init__(*args, **kwargs)

        ds = MIMIC_III()

        # map dtypes in all tables.
        # NOTE: necessary since duration times are bugged in pyarrow 14.
        for name, table in ds.tables.items():
            ds.tables[name] = map_pandas_arrowtime_numpy(table)

        # Preprocessing
        admissions = ds.ADMISSIONS
        patients = ds.PATIENTS
        static_covariates = pd.merge(admissions, patients, on="SUBJECT_ID")
        static_covariates = static_covariates.assign(
            ELAPSED_TIME=static_covariates.DISCHTIME - static_covariates.ADMITTIME
        )
        # select patients with unique ID
        counts = static_covariates["SUBJECT_ID"].value_counts()
        unique_patients = counts[counts == 1].index
        static_covariates = static_covariates.loc[
            static_covariates["SUBJECT_ID"].isin(unique_patients)
        ].reset_index(drop=True)
        # select patients with 2-30 days of data.
        # NOTE: Code by GRU-ODE-Bayes used `ELAPSED_TIME.dt.day > 2` but this is incorrect,
        #  because it will select patients with at least 72 hours of data.
        static_covariates = static_covariates.loc[
            (static_covariates.ELAPSED_TIME >= "2d")
            & (static_covariates.ELAPSED_TIME <= "30d")
        ]
        # select patients with age between 15 and 100 years at admission.
        static_covariates = static_covariates.assign(
            AGE=static_covariates.ADMITTIME - static_covariates.DOB
        )
        age = static_covariates.AGE
        year = np.timedelta64(365, "D")
        static_covariates = static_covariates.loc[
            (age >= 15 * year) & (age <= 100 * year)
        ]
        # select patients with "chartevents" data.
        static_covariates = static_covariates.loc[
            static_covariates.HAS_CHARTEVENTS_DATA
        ]

        # select relevant columns.
        static_covariates = static_covariates[
            [
                "SUBJECT_ID",
                "HADM_ID",
                "ADMITTIME",
                "DISCHTIME",
                "AGE",
                "ETHNICITY",
                "GENDER",
                "INSURANCE",
                "MARITAL_STATUS",
                "RELIGION",
            ]
        ]
