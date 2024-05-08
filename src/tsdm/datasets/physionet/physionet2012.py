r"""Physionet Challenge 2012.

Physionet Challenge 2012 Data Set
=================================

The development of methods for prediction of mortality rates in Intensive Care Unit (ICU) populations has been
motivated primarily by the need to compare the efficacy of medications, care guidelines, surgery, and other
interventions when, as is common, it is necessary to control for differences in severity of illness or trauma, age,
and other factors. For example, comparing overall mortality rates between trauma units in a community hospital,
a teaching hospital, and a military field hospital is likely to reflect the differences in the patient populations more
than any differences in standards of care. Acuity scores such as APACHE and SAPS-II are widely used to account for
these differences in the context of such studies.

By contrast, the focus of the PhysioNet/CinC Challenge 2012 is to develop methods for patient-specific prediction of in
hospital mortality. Participants will use information collected during the first two days of an ICU stay to predict
which patients survive their hospitalizations, and which patients do not.

Data for the Challenge
----------------------

The data used for the challenge consist of records from 12,000 ICU stays. All patients were
adults who were admitted for a wide variety of reasons to cardiac, medical, surgical, and
trauma ICUs. ICU stays of less than 48 hours have been excluded. Patients with DNR
(do not resuscitate) or CMO (comfort measures only) directives were not excluded.

Four thousand records comprise training set A, and the remaining records form test sets B and C.
Outcomes are provided for the training set records, and are withheld for the test set records.

Up to 42 variables were recorded at least once during the first 48 hours after admission
to the ICU. Not all variables are available in all cases, however. Six of these variables
are general descriptors (collected on admission), and the remainder are time series,
for which multiple observations may be available.

Each observation has an associated time-stamp indicating the elapsed time of the observation since
ICU admission in each case, in hours and minutes. Thus, for example, a time stamp of 35:19 means that
the associated observation was made 35 hours and 19 minutes after the patient was admitted to the ICU.

Each record is stored as a comma-separated value (CSV) text file. To simplify downloading, participants may download
a zip file or tarball containing all of training set A or test set B. Test set C will be used for validation only and
will not be made available to participants.


Update (8 May 2012): The extraneous ages that were present in the previous versions of some data files have been
removed, and a new general descriptor (ICUType, see below) has been added in each data file.

Five additional outcome-related descriptors, described below, are known for each record.
These are stored in separate CSV text files for each of sets A, B, and C, but only those for set A are available to
challenge participants.

All valid values for general descriptors, time series variables, and outcome-related descriptors are non-negative.
A value of -1 indicates missing or unknown data (for example, if a patient's height was not recorded).

General descriptors
-------------------

As noted, these six descriptors are collected at the time the patient is admitted to the ICU.
Their associated time-stamps are set to 00:00 (thus they appear at the beginning of each patient's record).

RecordID (a unique integer for each ICU stay)
Age (years)
Gender (0: female, or 1: male)
Height (cm)
ICUType (1: Coronary Care Unit, 2: Cardiac Surgery Recovery Unit, 3: Medical ICU, or 4: Surgical ICU)
Weight (kg)*.
The ICUType was added for use in Phase 2; it specifies the type of ICU to which the patient has been admitted.

Time Series
-----------

These 37 variables may be observed once, more than once, or not at all in some cases:

- Albumin (g/dL)
- ALP [Alkaline phosphatase (IU/L)]
- ALT [Alanine transaminase (IU/L)]
- AST [Aspartate transaminase (IU/L)]
- Bilirubin (mg/dL)
- BUN [Blood urea nitrogen (mg/dL)]
- Cholesterol (mg/dL)
- Creatinine [Serum creatinine (mg/dL)]
- DiasABP [Invasive diastolic arterial blood pressure (mmHg)]
- FiO2 [Fractional inspired O2 (0-1)]
- GCS [Glasgow Coma Score (3-15)]
- Glucose [Serum glucose (mg/dL)]
- HCO3 [Serum bicarbonate (mmol/L)]
- HCT [Hematocrit (%)]
- HR [Heart rate (bpm)]
- K [Serum potassium (mEq/L)]
- Lactate (mmol/L)
- Mg [Serum magnesium (mmol/L)]
- MAP [Invasive mean arterial blood pressure (mmHg)]
- MechVent [Mechanical ventilation respiration (0:false, or 1:true)]
- Na [Serum sodium (mEq/L)]
- NIDiasABP [Non-invasive diastolic arterial blood pressure (mmHg)]
- NIMAP [Non-invasive mean arterial blood pressure (mmHg)]
- NISysABP [Non-invasive systolic arterial blood pressure (mmHg)]
- PaCO2 [partial pressure of arterial CO2 (mmHg)]
- PaO2 [Partial pressure of arterial O2 (mmHg)]
- pH [Arterial pH (0-14)]
- Platelets (cells/nL)
- RespRate [Respiration rate (bpm)]
- SaO2 [O2 saturation in hemoglobin (%)]
- SysABP [Invasive systolic arterial blood pressure (mmHg)]
- Temp [Temperature (°C)]
- TropI [Troponin-I (μg/L)]
- TropT [Troponin-T (μg/L)]
- Urine [Urine output (mL)]
- WBC [White blood cell count (cells/nL)]
- Weight (kg)*

The time series measurements are recorded in chronological order within each record, and the associated time stamps
indicate the elapsed time since admission to the ICU. Measurements may be recorded at regular intervals ranging from
hourly to daily, or at irregular intervals as required. Not all time series are available in all cases.

In a few cases, such as blood pressure, different measurements made using two or more methods or sensors
may be recorded with the same or only slightly different time-stamps. Occasional outliers should be expected as well.

Note that Weight is both a general descriptor (recorded on admission) and a time series variable
(often measured hourly, for estimating fluid balance).

Outcome-related Descriptors
---------------------------

The outcome-related descriptors are kept in a separate CSV text file for each of the three record sets; as noted, only
the file associated with training set A is available to participants. Each line of the outcomes file contains these
descriptors:

- RecordID (defined as above)
- SAPS-I score (Le Gall et al., 1984)
- SOFA score (Ferreira et al., 2001)
- Length of stay (days)
- Survival (days)
- In-hospital death (0: survivor, or 1: died in-hospital)

The Length of stay is the number of days between the patient's admission to the ICU and the end of hospitalization
(including any time spent in the hospital after discharge from the ICU).
If the patient's death was recorded (in or out of hospital), then Survival is the number of days between ICU admission
and death; otherwise, Survival is assigned the value -1. Since patients who spent less than 48 hours in the ICU have
been excluded, Length of stay and Survival never have the values 0 or 1 in the challenge data sets.
Given these definitions and constraints,

- Survival > Length of stay  =>  Survivor
- Survival = -1  =>  Survivor
- 2 <= Survival <= Length of stay  =>  In-hospital death
"""

__all__ = [
    # Constants
    "TIMESERIES_DESCRIPTION",
    "METADATA_DESCRIPTION",
    # Classes
    "PhysioNet2012",
]

import tarfile

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm.autonotebook import tqdm
from typing_extensions import Literal, TypeAlias

from tsdm.data import InlineTable, make_dataframe, remove_outliers
from tsdm.datasets.base import MultiTableDataset

KEY: TypeAlias = Literal[
    "timeseries",
    "timeseries_description",
    "metadata",
    "metadata_description",
    "raw_timeseries",
    "raw_metadata",
]

TIMESERIES_DESCRIPTION: InlineTable = {
    "data": [
        # variable, lower, upper, lower_included, upper_included, unit, description
        ("Albumin"    , 0,    None, False, True, "g/dL",     None                                            ),
        ("ALP"        , 0,    None, False, True, "IU/L",     "Alkaline phosphatase"                          ),
        ("ALT"        , 0,    None, False, True, "IU/L",     "Alanine transaminase"                          ),
        ("AST"        , 0,    None, False, True, "IU/L",     "Aspartate transaminase"                        ),
        ("Bilirubin"  , 0,    None, False, True, "mg/dL",    "Bilirubin"                                     ),
        ("BUN"        , 0,    None, False, True, "mg/dL",    "BUN"                                           ),
        ("Cholesterol", 0,    None, False, True, "mg/dL",    None                                            ),
        ("Creatinine" , 0,    None, False, True, "mg/dL",    "Serum creatinine"                              ),
        ("DiasABP"    , 0,    None, False, True, "mmHg",     "Invasive diastolic arterial blood pressure"    ),
        ("FiO2"       , 0,    1,    True,  True, "0-1",      "Fractional inspired O2"                        ),
        ("GCS"        , 3,    15,   True,  True, "3-15",     "Glasgow Coma Score "                           ),
        ("Glucose"    , 0,    None, False, True, "mg/dL",    "Serum glucose"                                 ),
        ("HCO3"       , 0,    None, False, True, "mmol/L",   "Serum bicarbonate"                             ),
        ("HCT"        , 0,    100,  True,  True, "%",        "Hematocrit"                                    ),
        ("HR"         , 0,    None, True,  True, "bpm",      "Heart rate"                                    ),
        ("K"          , 0,    None, False, True, "mEq/L",    "Serum potassium"                               ),
        ("Lactate"    , 0,    None, False, True, "mmol/L",   None                                            ),
        ("Mg"         , 0,    None, False, True, "mmol/L",   "Serum magnesium"                               ),
        ("MAP"        , 0,    None, False, True, "mmHg",     "Invasive mean arterial blood pressure"         ),
        ("MechVent"   , None, None, True,  True, "bool",     "Mechanical ventilation respiration"            ),
        ("Na"         , 0,    None, False, True, "mEq/L",    "Serum sodium"                                  ),
        ("NIDiasABP"  , 0,    None, False, True, "mmHg",     "Non-invasive diastolic arterial blood pressure"),
        ("NIMAP"      , 0,    None, False, True, "mmHg",     "Non-invasive mean arterial blood pressure"     ),
        ("NISysABP"   , 0,    None, False, True, "mmHg",     "Non-invasive systolic arterial blood pressure" ),
        ("PaCO2"      , 0,    None, False, True, "mmHg",     "partial pressure of arterial CO2"              ),
        ("PaO2"       , 0,    None, False, True, "mmHg",     "Partial pressure of arterial O2"               ),
        ("pH"         , 0,    14,   False, True, "0-14",     "Arterial pH"                                   ),
        ("Platelets"  , 0,    None, False, True, "cells/nL", None                                            ),
        ("RespRate"   , 0,    None, True, True,  "bpm",      "Respiration rate"                              ),
        ("SaO2"       , 0,    100,  True,  True, "%", "      O2 saturation in hemoglobin"                    ),
        ("SysABP"     , 0,    None, False, True, "mmHg",     "Invasive systolic arterial blood pressure"     ),
        ("Temp"       , 0,    None, False, True, "°C",       "Temperature"                                   ),
        ("TroponinI"  , 0,    None, False, True, "μg/L",     "Troponin-I"                                    ),
        ("TroponinT"  , 0,    None, False, True, "μg/L",     "Troponin-T"                                    ),
        ("Urine"      , 0,    None, True,  True, "mL",       "Urine output"                                  ),
        ("WBC"        , 0,    1000, False, True, "cells/nL", "White blood cell count"                        ),
        ("Weight"     , 20,   None, True,  True, "kg",       None                                            ),
    ],
    "schema": {
        "name"            : "string[pyarrow]",
        "lower_bound"     : "float32[pyarrow]",
        "upper_bound"     : "float32[pyarrow]",
        "lower_inclusive" : "bool[pyarrow]",
        "upper_inclusive" : "bool[pyarrow]",
        "unit"            : "string[pyarrow]",
        "description"     : "string[pyarrow]",
    },
    "index": ["name"],
}  # fmt: skip

METADATA_DESCRIPTION: InlineTable = {
    "data": [
        ("Age"    , "uint8[pyarrow]"  , 0   , 100 , True, True, "years"   , None                           ),
        ("Gender" , "int8[pyarrow]"   , None, None, True, True, "category", "0: female, 1: male, -1: other"),
        ("Height" , "float32[pyarrow]", 20  , 270 , True, True, "cm"      , None                           ),
        ("Weight" , "float32[pyarrow]", 20  , None, True, True, "kg"      , None                           ),
        ("ICUType", "uint8[pyarrow]"  , 1   , 4   , True, True, "category",
            "1: Coronary Care Unit, 2: Cardiac Surgery Recovery Unit, 3: Medical ICU, or 4: Surgical ICU"),
    ],
    "schema": {
        "name"            : "string[pyarrow]",
        "dtype"           : "string[pyarrow]",
        "lower_bound"     : "float32[pyarrow]",
        "upper_bound"     : "float32[pyarrow]",
        "lower_inclusive" : "bool[pyarrow]",
        "upper_inclusive" : "bool[pyarrow]",
        "unit"            : "string[pyarrow]",
        "description"     : "string[pyarrow]",
    },
    "index": ["name"],
}  # fmt: skip


class PhysioNet2012(MultiTableDataset[KEY, DataFrame]):
    r"""Physionet Challenge 2012.

    Each training data file provides two tables.
    The first table provides general descriptors of patients:

    +----------+-----+--------+--------+---------+--------+
    | RecordID | Age | Gender | Height | ICUType | Weight |
    +==========+=====+========+========+=========+========+
    | 141834   | 52  | 1.0    | 172.7  | 2       | 73.0   |
    +----------+-----+--------+--------+---------+--------+
    | 133786   | 46  | 0.0    | 157.5  | 1       | 52.3   |
    +----------+-----+--------+--------+---------+--------+
    | 141492   | 84  | 0.0    | 152.4  | 3       | 61.2   |
    +----------+-----+--------+--------+---------+--------+
    | 142386   | 87  | 0.0    | 160.0  | 4       | 73.0   |
    +----------+-----+--------+--------+---------+--------+
    | 142258   | 71  | 1.0    | NaN    | 3       | 72.9   |
    +----------+-----+--------+--------+---------+--------+
    | ...      | ... | ...    | ...    | ...     | ...    |
    +----------+-----+--------+--------+---------+--------+
    | 142430   | 39  | 0.0    | 157.5  | 2       | 65.9   |
    +----------+-----+--------+--------+---------+--------+
    | 134614   | 77  | 0.0    | 165.1  | 1       | 66.6   |
    +----------+-----+--------+--------+---------+--------+
    | 139802   | 57  | 1.0    | NaN    | 4       | NaN    |
    +----------+-----+--------+--------+---------+--------+
    | 136653   | 57  | 1.0    | NaN    | 3       | 103.9  |
    +----------+-----+--------+--------+---------+--------+
    | 136047   | 67  | 1.0    | NaN    | 3       | 169.0  |
    +----------+-----+--------+--------+---------+--------+

    where `RecordID` defines unique ID of an admission.

    The second table contains measurements over time, each column of the table provides
    a sequence of measurements over time (e.g., arterial pH), where the
    header of the column describes the measurement. Each row of the table provides a collection of
    measurements at the same time (e.g., heart rate and oxygen level at the same time) for the same admission.

    The table is formatted in the following way:

    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    | RecordID | Time | BUN | Creatinine | DiasABP | ... | Cholesterol | TroponinT | TroponinI |
    +==========+======+=====+============+=========+=====+=============+===========+===========+
    | 141834   | 27   | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 107  | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 112  | NaN | NaN        | 77.0    | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 127  | NaN | NaN        | 81.0    | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 142  | NaN | NaN        | 74.0    | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    | ...      |      | ... | ...        | ...     | ... | ...         | ...       | ...       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    | 136047   | 2618 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 2678 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 2738 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+
    |          | 2798 | NaN | NaN        | NaN     | ... | NaN         | NaN       | NaN       |
    +----------+------+-----+------------+---------+-----+-------------+-----------+-----------+

    Entries of NaN (not a number) indicate that there was no recorded measurement of a variable at the time.
    """

    SOURCE_URL = r"https://archive.physionet.org/challenge/2012/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = r"https://archive.physionet.org/challenge/2012/"
    r"""HTTP address containing additional information about the dataset."""

    rawdata_files = ["set-a.tar.gz", "set-b.tar.gz", "set-c.tar.gz"]

    rawdata_hashes = {
        "set-a.tar.gz": "sha256:8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920",
        "set-b.tar.gz": "sha256:b1637a2a423a8e76f8f087896cfc5fdf28f88519e1f4e874fbda69b2a64dac30",
        "set-c.tar.gz": "sha256:a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1",
    }

    rawdata_schema = {
        "Time": "string[pyarrow]",
        "Parameter": "string[pyarrow]",
        "Value": "float32[pyarrow]",
    }

    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "metadata",
        "timeseries_description",
        "metadata_description",
        "raw_timeseries",
        "raw_metadata",
    ]

    table_schemas = {# pyright: ignore[reportAssignmentType]
        "timeseries": {
            "Albumin"     : "float32[pyarrow]",
            "ALP"         : "float32[pyarrow]",
            "ALT"         : "float32[pyarrow]",
            "AST"         : "float32[pyarrow]",
            "Bilirubin"   : "float32[pyarrow]",
            "BUN"         : "float32[pyarrow]",
            "Cholesterol" : "float32[pyarrow]",
            "Creatinine"  : "float32[pyarrow]",
            "DiasABP"     : "float32[pyarrow]",
            "FiO2"        : "float32[pyarrow]",
            "GCS"         : "float32[pyarrow]",
            "Glucose"     : "float32[pyarrow]",
            "HCO3"        : "float32[pyarrow]",
            "HCT"         : "float32[pyarrow]",
            "HR"          : "float32[pyarrow]",
            "K"           : "float32[pyarrow]",
            "Lactate"     : "float32[pyarrow]",
            "MAP"         : "float32[pyarrow]",
            "MechVent"    : "float32[pyarrow]",
            "Mg"          : "float32[pyarrow]",
            "Na"          : "float32[pyarrow]",
            "NIDiasABP"   : "float32[pyarrow]",
            "NIMAP"       : "float32[pyarrow]",
            "NISysABP"    : "float32[pyarrow]",
            "PaCO2"       : "float32[pyarrow]",
            "PaO2"        : "float32[pyarrow]",
            "pH"          : "float32[pyarrow]",
            "Platelets"   : "float32[pyarrow]",
            "RespRate"    : "float32[pyarrow]",
            "SaO2"        : "float32[pyarrow]",
            "SysABP"      : "float32[pyarrow]",
            "Temp"        : "float32[pyarrow]",
            "TroponinI"   : "float32[pyarrow]",
            "TroponinT"   : "float32[pyarrow]",
            "Urine"       : "float32[pyarrow]",
            "WBC"         : "float32[pyarrow]",
            "Weight"      : "float32[pyarrow]",
        },
        "metadata": {
            "Age"     : "uint8[pyarrow]",
            "Gender"  : "int8[pyarrow]",
            "Height"  : "float32[pyarrow]",
            "ICUType" : "uint8[pyarrow]",
            "Weight"  : "float32[pyarrow]",
        },
        "timeseries_description": TIMESERIES_DESCRIPTION["schema"],
        "metadata_description": METADATA_DESCRIPTION["schema"],
    }  # fmt: skip

    def _clean_single_rawdataset(self, fname: str) -> tuple[DataFrame, DataFrame]:
        with (
            tarfile.open(self.rawdata_paths[fname], "r") as archive,
            tqdm(archive.getmembers()) as progress_bar,
        ):
            progress_bar.set_description(f"Loading patient data from {fname}")
            id_list = []
            md_list = []
            ts_list = []

            for member in progress_bar:
                if not member.isfile():
                    continue

                record_id = int("".join(c for c in member.name if c.isdigit()))
                progress_bar.set_postfix(record_id=record_id)
                with archive.extractfile(member) as file:  # type: ignore[union-attr]
                    df = pd.read_csv(
                        file,
                        dtype=self.rawdata_schema,
                        dtype_backend="pyarrow",
                    )
                    assert record_id == int(df.iloc[0, -1]), "RecordID mismatch!"
                    df = df.iloc[1:]

                    # drop rows if Parameter is NaN
                    df = df.dropna(subset=["Parameter"])

                    # select metadata items
                    md_mask = (df["Time"] == "00:00") & df["Parameter"].isin(
                        self.table_schemas["metadata"]
                    )
                    # keep the first instance of each metadata item
                    md_mask &= ~df.loc[md_mask, "Parameter"].duplicated()
                    md_frame = df.loc[md_mask].drop(columns=["Time"])
                    assert len(md_frame) <= 5, "Too many metadata items!"

                    ts_frame = df.loc[~md_mask]  # remaining items
                    assert all(
                        ts_frame["Parameter"].isin(self.table_schemas["timeseries"])
                    )
                    id_list.append(record_id)
                    md_list.append(md_frame)
                    ts_list.append(ts_frame)

        record_ids = pd.Series(id_list, name="RecordID")

        self.LOGGER.info("%s: Combining metadata.", fname)
        md = pd.concat(md_list, axis=0, keys=record_ids).reset_index(
            level=-1, drop=True
        )

        self.LOGGER.info("%s: Performing pivot on metadata.", fname)
        md = md.pivot(columns="Parameter", values="Value").astype(
            self.table_schemas["metadata"]
        )

        self.LOGGER.info("%s: Combining timeseries data.", fname)
        ts = pd.concat(ts_list, axis=0, keys=record_ids).reset_index(
            level=-1, drop=True
        )
        ts = ts.assign(  # from hh:mm to timedelta64
            Time=ts["Time"]
            .str.split(":", expand=True)
            .astype(int)
            .dot([3600, 60])
            .mul(np.timedelta64(1, "s"))
        )

        self.LOGGER.info("%s: Performing non-aggregating pivot.", fname)
        ts = (
            ts.reset_index()
            .set_index(
                ts.groupby(["RecordID", "Time", "Parameter"]).cumcount().rename("count")
            )
            .set_index(["RecordID", "Time", "Parameter"], append=True)
            .unstack(level="Parameter")
            .reset_index("count", drop=True)
            .droplevel(0, axis="columns")
            .sort_index()
            .reindex(columns=self.table_schemas["timeseries"])
        )
        return ts, md

    def _clean_all_rawdatasets(self) -> None:
        ts_list = []
        md_list = []
        for fname in self.rawdata_files:
            ts, md = self._clean_single_rawdataset(fname)
            ts_list.append(ts)
            md_list.append(md)
        ts = pd.concat(ts_list)
        md = pd.concat(md_list)
        # NOTE: TS is missing a few records, since no time series data was available
        # For tasks, it is recommended to drop records with less than 24 observations
        # assert md.index == ts.index.get_level_values("RecordID").unique()
        self.serialize_table(md, self.dataset_paths["raw_metadata"])
        self.serialize_table(ts, self.dataset_paths["raw_timeseries"])

    def clean_table(self, key: KEY) -> None | DataFrame:
        match key:
            case "timeseries_description":
                return make_dataframe(**TIMESERIES_DESCRIPTION)
            case "metadata_description":
                return make_dataframe(**METADATA_DESCRIPTION)
            case "timeseries":
                self.LOGGER.info("Removing outliers from timeseries.")
                ts = remove_outliers(self.raw_timeseries, self.timeseries_description)
                self.LOGGER.info("Dropping completely missing rows.")
                ts = ts.dropna(how="all", axis="index")
                return ts
            case "metadata":
                self.LOGGER.info("Removing outliers from metadata.")
                return remove_outliers(self.raw_metadata, self.metadata_description)
            case "raw_timeseries" | "raw_metadata":
                return self._clean_all_rawdatasets()
            case _:
                raise KeyError(f"Unknown table: {key!r} not in {self.table_names}")
