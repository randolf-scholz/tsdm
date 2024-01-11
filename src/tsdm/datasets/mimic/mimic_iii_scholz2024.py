"""Custom processed version of the MIMIC-III dataset."""

__all__ = ["MIMIC_III_Scholz2024"]

import numpy as np
import pandas as pd
import pyarrow as pa

import tsdm
from tsdm.datasets.mimic.mimic_iii import MIMIC_III

ARROW_DURATION_TYPES = {
    pd.ArrowDtype(pa.duration(unit)) for unit in ["s", "ms", "us", "ns"]
}

ARROW_TIMESTAMP_TYPES = {
    pd.ArrowDtype(pa.timestamp(unit)) for unit in ["s", "ms", "us", "ns"]
}

ARROW_DATE_TYPES = {pd.ArrowDtype(pa.date32()), pd.ArrowDtype(pa.date64())}


def map_dtypes(df):
    """Converts pyarrow date/timestamp/duration types to numpy equivalents.

    Rationale: pyarrow types are currently bugged and do not support all operations.
    """
    for col, dtype in df.dtypes.items():
        if dtype in ARROW_DURATION_TYPES:
            df[col] = df[col].astype("timedelta64[ms]")
        elif dtype in ARROW_TIMESTAMP_TYPES:
            df[col] = df[col].astype("datetime64[ms]")
        elif dtype in ARROW_DATE_TYPES:
            df[col] = df[col].astype("datetime64[s]")
    return df


class MIMIC_III_Scholz2024(MIMIC_III):
    """Custom processed version of the MIMIC-III dataset."""

    RAWDATA_DIR = MIMIC_III.RAWDATA_DIR

    def __init__(self, *args, **kwargs):
        """Initialize the dataset."""
        super().__init__(*args, **kwargs)

        ds = tsdm.datasets.MIMIC_III()

        # map dtypes in all tables.
        # NOTE: necessary since duration times are bugged in pyarrow 14.
        for name, table in ds.tables.items():
            ds.tables[name] = map_dtypes(table)

        # Preprocessing
        admissions = ds.ADMISSIONS
        patients = ds.PATIENTS
        metadata = pd.merge(admissions, patients, on="SUBJECT_ID")
        metadata = metadata.assign(ELAPSED_TIME=metadata.DISCHTIME - metadata.ADMITTIME)
        # select patients with unique ID
        counts = metadata["SUBJECT_ID"].value_counts()
        unique_patients = counts[counts == 1].index
        metadata = metadata.loc[
            metadata["SUBJECT_ID"].isin(unique_patients)
        ].reset_index(drop=True)
        # select patients with 2-30 days of data.
        # NOTE: Code by GRU-ODE-Bayes used `ELAPSED_TIME.dt.day > 2` but this is incorrect,
        #  because it will select patients with at least 72 hours of data.
        metadata = metadata.loc[
            (metadata.ELAPSED_TIME >= "2d") & (metadata.ELAPSED_TIME <= "30d")
        ]
        # select patients with age between 15 and 100 years at admission.
        YEAR = np.timedelta64(365, "D")
        metadata = metadata.assign(AGE=metadata.ADMITTIME - metadata.DOB)
        metadata = metadata.loc[
            (metadata.AGE >= 15 * YEAR) & (metadata.AGE <= 100 * YEAR)
        ]
        # select patients with chartevents data.
        metadata = metadata.loc[metadata.HAS_CHARTEVENTS_DATA]

        # select relevant columns.
        metadata = metadata[[
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
        ]]
