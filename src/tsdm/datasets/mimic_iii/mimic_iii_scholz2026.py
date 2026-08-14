r"""Custom processed version of the MIMIC-III dataset."""

__all__ = [
    "BOOL_VALUES",
    "NULL_VALUES",
    # Classes
    "MIMIC_III_Scholz2026",
]

from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from tsdm.backend.pyarrow import cast_columns, filter_nulls, set_nulls
from tsdm.datatools import strip_whitespace
from tsdm.dtypes import map_pandas_arrowtime_numpy

from .mimic_iii import MIMIC_III, MIMIC_III_Key

BOOL_VALUES: dict[MIMIC_III_Key, dict[str, dict[Any, bool]]] = {
    "ADMISSIONS": {
        "HOSPITAL_EXPIRE_FLAG": {0: False, 1: True},
        "HAS_CHARTEVENTS_DATA": {0: False, 1: True},
    },
    "CALLOUT": {
        "REQUEST_TELE": {0: False, 1: True},
        "REQUEST_RESP": {0: False, 1: True},
        "REQUEST_CDIFF": {0: False, 1: True},
        "REQUEST_MRSA": {0: False, 1: True},
        "REQUEST_VRE": {0: False, 1: True},
    },
    "CHARTEVENTS": {
        "WARNING": {0: False, 1: True},
        "ERROR": {0: False, 1: True},
    },
    "DATETIMEEVENTS": {
        "WARNING": {0: False, 1: True},
        "ERROR": {0: False, 1: True},
    },
    "D_CPT": {"CODESUFFIX": {"F": False, "T": True}},
    "INPUTEVENTS_CV": {"NEWBOTTLE": {0: False, 1: True}},
    "INPUTEVENTS_MV": {
        "ISOPENBAG": {0: False, 1: True},
        "CONTINUEINNEXTDEPT": {0: False, 1: True},
    },
    "NOTEEVENTS": {"ISERROR": {0: False, 1: True}},
    "PATIENTS": {"EXPIRE_FLAG": {0: False, 1: True}},
    "PROCEDUREEVENTS_MV": {
        "ISOPENBAG": {0: False, 1: True},
        "CONTINUEINNEXTDEPT": {0: False, 1: True},
    },
}
NULL_VALUES = [
    "",
    " ",
    "  ",
    "   ",
    "    ",
    "     ",
    "      ",
    "       ",
    "        ",
    "-",
    "---",
    "----",
    "-----",
    "-------",
    "?",
    "UNABLE TO OBTAIN",
    "UNKNOWN",
    "Unknown",
    "unknown",
    ".",
    ".*.",
    "___.",
    "_",
    "__",
    "___",
]


class MIMIC_III_Scholz2026(MIMIC_III):
    r"""Custom processed version of the MIMIC-III dataset."""

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
        static_covariates = pd.merge(admissions, patients, on="SUBJECT_ID")  # ruff: ignore[PD015]
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

    def __post_init__(self) -> None:
        # Reuse the raw data and lazily materialized raw tables.
        self.raw_dataset = MIMIC_III(
            version=self.__version__, initialize=False, verbose=self.verbose
        )
        self.RAWDATA_DIR = self.raw_dataset.RAWDATA_DIR

    def clean_table(self, key: MIMIC_III_Key) -> pa.Table:
        self.raw_dataset.clean(key, validate_rawdata=False)
        table = self.raw_dataset.load_table(key).collect().to_arrow()

        # Post-processing
        match key:
            case "ADMISSIONS":
                table = set_nulls(
                    table,
                    ETHNICITY=["UNKNOWN/NOT SPECIFIED"],
                    RELIGION=["NOT SPECIFIED", "UNOBTAINABLE"],
                    MARITAL_STATUS=["UNKNOWN (DEFAULT)"],
                )
            case "CALLOUT":
                pass
            case "CAREGIVERS":
                pass
            case "CHARTEVENTS":
                table = filter_nulls(
                    table, "ICUSTAY_ID", "VALUE", "VALUENUM", "VALUEUOM"
                )
                table = cast_columns(table, VALUE="float64")
            case "CPTEVENTS":
                table = cast_columns(table, CHARTDATE="date32")
            case "DATETIMEEVENTS":
                pass
            case "DIAGNOSES_ICD":
                pass
            case "DRGCODES":
                pass
            case "D_CPT":
                pass
            case "D_ICD_DIAGNOSES":
                pass
            case "D_ICD_PROCEDURES":
                pass
            case "D_ITEMS":
                pass
            case "D_LABITEMS":
                pass
            case "ICUSTAYS":
                pass
            case "INPUTEVENTS_CV":
                pass
            case "INPUTEVENTS_MV":
                pass
            case "LABEVENTS":
                table = filter_nulls(table, "VALUE", "VALUENUM", "VALUEUOM")
                table = strip_whitespace(table)
                table = cast_columns(table, VALUE="float64")
            case "MICROBIOLOGYEVENTS":
                table = cast_columns(table, CHARTDATE="date32")
            case "NOTEEVENTS":
                pass
            case "OUTPUTEVENTS":
                table = filter_nulls(table, "VALUE", "VALUEUOM")
                table = cast_columns(table, VALUE="float64")
            case "PATIENTS":
                table = cast_columns(
                    table,
                    DOB="date32",
                    DOD="date32",
                    DOD_HOSP="date32",
                    DOD_SSN="date32",
                )
            case "PRESCRIPTIONS":
                table = cast_columns(table, STARTDATE="date32", ENDDATE="date32")
            case "PROCEDUREEVENTS_MV":
                pass
            case "PROCEDURES_ICD":
                pass
            case "SERVICES":
                pass
            case "TRANSFERS":
                pass
            case _:
                raise ValueError(f"Unknown table name: {key}")

        return table

    def load_table(self, key: MIMIC_III_Key, /) -> pa.Table:
        return pq.read_table(self.dataset_paths[key])
