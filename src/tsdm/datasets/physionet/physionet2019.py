r"""Physionet Challenge 2019.

Physionet Challenge 2019 Data Set
=================================

The Challenge data repository contains one file per subject (e.g. training/p00101.psv for the
training data). The complete training database (42 MB) consists of two parts: training set A
(20,336 subjects) and B (20,000 subjects).

Each training data file provides a table with measurements over time. Each column of the table
provides a sequence of measurements over time (e.g., heart rate over several hours), where the
header of the column describes the measurement. Each row of the table provides a collection of
measurements at the same time (e.g., heart rate and oxygen level at the same time).

The table is formatted in the following way:

+-----+-------+------+-----+-------------+--------+-------------+
| HR  | O2Sat | Temp | ... | HospAdmTime | ICULOS | SepsisLabel |
+=====+=======+======+=====+=============+========+=============+
| NaN | NaN   | NaN  | ... | -50         | 1      | 0           |
+-----+-------+------+-----+-------------+--------+-------------+
| 86  | 98    | NaN  | ... | -50         | 2      | 0           |
+-----+-------+------+-----+-------------+--------+-------------+
| 75  | NaN   | NaN  | ... | -50         | 3      | 1           |
+-----+-------+------+-----+-------------+--------+-------------+
| 99  | 100   | 35.5 | ... | -50         | 4      | 1           |
+-----+-------+------+-----+-------------+--------+-------------+

There are 40 time-dependent variables HR, O2Sat, Temp ..., HospAdmTime, which are described here.
The final column, SepsisLabel, indicates the onset of sepsis according to the Sepsis-3 definition,
where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number) indicate that
there was no recorded measurement of a variable at the time interval.

More details
------------

Data used in the competition is sourced from ICU patients in three separate hospital systems.
Data from two hospital systems will be publicly available; however, one data set will be censored
and used for scoring. The data for each patient will be contained within a single pipe-delimited
text file. Each file will have the same header, and each row will represent a single hour's worth
of data. Available patient co-variates consist of Demographics, Vital Signs, and Laboratory values,
which are defined in the tables below.

The following time points are defined for each patient:

tsuspicion

    1. Clinical suspicion of infection, identified as the earlier timestamp of IV antibiotics and
       blood cultures within a specified duration.
    2. If antibiotics were given first, then the cultures must have been obtained within 24 hours.
       If cultures were obtained first, then antibiotics must have been subsequently ordered within
       72 hours.
    3. Antibiotics must have been administered for at least 72 consecutive hours to be considered.

tSOFA

    The occurrence of end organ damage as identified by a two-point deterioration in SOFA score
    within a 24h period.

tsepsis

    The onset time of sepsis is the earlier of tsuspicion and tSOFA as long as tSOFA occurs no more
    than 24 hours before or 12 hours after tsuspicion; otherwise, the patient is not marked as a
    sepsis patient. Specifically, if $t_{\text{suspicion}}−24 ≤ t_{\text{SOFA}} ≤ t_{\text{suspicion}}+12$,
    then $t_{\text{sepsis}} = \min(t_{\text{suspicion}}, t_{\text{SOFA}})$.

Table 1: Columns in each training data file. Vital signs (columns 1-8)
HR 	Heart rate (beats per minute)

+------------------+------------------------------------------------------------------+
| O2Sat            | Pulse oximetry (%)                                               |
+==================+==================================================================+
| Temp             | Temperature (Deg C)                                              |
+------------------+------------------------------------------------------------------+
| SBP              | Systolic BP (mm Hg)                                              |
+------------------+------------------------------------------------------------------+
| MAP              | Mean arterial pressure (mm Hg)                                   |
+------------------+------------------------------------------------------------------+
| DBP              | Diastolic BP (mm Hg)                                             |
+------------------+------------------------------------------------------------------+
| Resp             | Respiration rate (breaths per minute)                            |
+------------------+------------------------------------------------------------------+
| EtCO2            | End tidal carbon dioxide (mm Hg)                                 |
+------------------+------------------------------------------------------------------+
| Laboratory       | values (columns 9-34)                                            |
+------------------+------------------------------------------------------------------+
| BaseExcess       | Measure of excess bicarbonate (mmol/L)                           |
+------------------+------------------------------------------------------------------+
| HCO3             | Bicarbonate (mmol/L)                                             |
+------------------+------------------------------------------------------------------+
| FiO2             | Fraction of inspired oxygen (%)                                  |
+------------------+------------------------------------------------------------------+
| pH               | N/A                                                              |
+------------------+------------------------------------------------------------------+
| PaCO2            | Partial pressure of carbon dioxide from arterial blood (mm Hg)   |
+------------------+------------------------------------------------------------------+
| SaO2             | Oxygen saturation from arterial blood (%)                        |
+------------------+------------------------------------------------------------------+
| AST              | Aspartate transaminase (IU/L)                                    |
+------------------+------------------------------------------------------------------+
| BUN              | Blood urea nitrogen (mg/dL)                                      |
+------------------+------------------------------------------------------------------+
| Alkalinephos     | Alkaline phosphatase (IU/L)                                      |
+------------------+------------------------------------------------------------------+
| Calcium          | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Chloride         | (mmol/L)                                                         |
+------------------+------------------------------------------------------------------+
| Creatinine       | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Bilirubin_direct | Bilirubin direct (mg/dL)                                         |
+------------------+------------------------------------------------------------------+
| Glucose          | Serum glucose (mg/dL)                                            |
+------------------+------------------------------------------------------------------+
| Lactate          | Lactic acid (mg/dL)                                              |
+------------------+------------------------------------------------------------------+
| Magnesium        | (mmol/dL)                                                        |
+------------------+------------------------------------------------------------------+
| Phosphate        | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Potassium        | (mmol/L)                                                         |
+------------------+------------------------------------------------------------------+
| Bilirubin_total  | Total bilirubin (mg/dL)                                          |
+------------------+------------------------------------------------------------------+
| TroponinI        | Troponin I (ng/mL)                                               |
+------------------+------------------------------------------------------------------+
| Hct              | Hematocrit (%)                                                   |
+------------------+------------------------------------------------------------------+
| Hgb              | Hemoglobin (g/dL)                                                |
+------------------+------------------------------------------------------------------+
| PTT              | partial thromboplastin time (seconds)                            |
+------------------+------------------------------------------------------------------+
| WBC              | Leukocyte count (count*10^3/µL)                                  |
+------------------+------------------------------------------------------------------+
| Fibrinogen       | (mg/dL)                                                          |
+------------------+------------------------------------------------------------------+
| Platelets        | (count*10^3/µL)                                                  |
+------------------+------------------------------------------------------------------+
| Demographics     | (columns 35-40)                                                  |
+------------------+------------------------------------------------------------------+
| Age              | Years (100 for patients 90 or above)                             |
+------------------+------------------------------------------------------------------+
| Gender           | Female (0) or Male (1)                                           |
+------------------+------------------------------------------------------------------+
| Unit1            | Administrative identifier for ICU unit (MICU)                    |
+------------------+------------------------------------------------------------------+
| Unit2            | Administrative identifier for ICU unit (SICU)                    |
+------------------+------------------------------------------------------------------+
| HospAdmTime      | Hours between hospital admit and ICU admit                       |
+------------------+------------------------------------------------------------------+
| ICULOS           | ICU length-of-stay (hours since ICU admit)                       |
+------------------+------------------------------------------------------------------+
| Outcome          | (column 41)                                                      |
+------------------+------------------------------------------------------------------+
| SepsisLabel      | For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6 and         |
|                  | 0 if t<tsepsis−6. For non-sepsis patients, SepsisLabel is 0.     |
+------------------+------------------------------------------------------------------+
"""

__all__ = [
    # Constants
    "RAWDATA_SCHEMA",
    "STATIC_COVARIATES_METADATA",
    "STATIC_COVARIATES_METADATA_SCHEMA",
    "STATIC_COVARIATES_SCHEMA",
    "TIMESERIES_METADATA",
    "TIMESERIES_METADATA_SCHEMA",
    "TIMESERIES_SCHEMA",
    # Classes
    "PhysioNet2019",
]

import re
from pathlib import Path
from typing import Literal, Optional
from zipfile import ZipFile

import polars as pl
from tqdm.auto import tqdm

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import remove_outliers
from tsdm.testing.hashutils import hash_zip_contents
from tsdm.testing.validation import ErrorHandler, validate_hash
from tsdm.utils import remote

TIMESERIES_METADATA = [
    ("patient",          None, None, None, None, None,      "Patient identifier"                         ),
    ("time",             None, None, None, None, "hours",   "ICU length of stay"                         ),
    ("HR",               0,    None, True, True, "bpm",     "Heart rate"                                 ),
    ("O2Sat",            0,    100,  True, True, "%",       "Pulse oximetry"                             ),
    ("Temp",             0,    None, True, True, "℃",       "Temperature"                                ),
    ("SBP",              0,    None, True, True, "mm Hg",   "Systolic BP"                                ),
    ("MAP",              0,    None, True, True, "mm Hg",   "Mean arterial pressure"                     ),
    ("DBP",              0,    None, True, True, "mm Hg",   "Diastolic BP"                               ),
    ("Resp",             0,    None, True, True, "bpm",     "Respiration rate"                           ),
    ("EtCO2",            0,    None, True, True, "mm Hg",   "End tidal carbon dioxide"                   ),
    # Laboratory values (columns 9-34)
    ("BaseExcess",       None, None, True, True, "mmol/L",  "Measure of excess bicarbonate"              ),
    ("HCO3",             0,    None, True, True, "mmol/L",  "Bicarbonate"                                ),
    ("FiO2",             0,    100,  True, True, "%",       "Fraction of inspired oxygen"                ),
    ("pH",               0,    14,   True, True, "pH",      None                                         ),
    ("PaCO2",            0,    None, True, True, "mm Hg",   "Partial pressure of CO₂ from arterial blood"),
    ("SaO2",             0,    100,  True, True, "%",       "Oxygen saturation from arterial blood"      ),
    ("AST",              0,    None, True, True, "IU/L",    "Aspartate transaminase"                     ),
    ("BUN",              0,    None, True, True, "mg/dL",   "Blood urea nitrogen"                        ),
    ("Alkalinephos",     0,    None, True, True, "IU/L",    "Alkaline phosphatase"                       ),
    ("Calcium",          0,    None, True, True, "mg/dL",   None                                         ),
    ("Chloride",         0,    None, True, True, "mmol/L",  None                                         ),
    ("Creatinine",       0,    None, True, True, "mg/dL",   None                                         ),
    ("Bilirubin_direct", 0,    None, True, True, "mg/dL",   "Bilirubin direct"                           ),
    ("Glucose",          0,    None, True, True, "mg/dL",   "Serum glucose"                              ),
    ("Lactate",          0,    None, True, True, "mg/dL",   "Lactic acid"                                ),
    ("Magnesium",        0,    None, True, True, "mmol/dL", None                                         ),
    ("Phosphate",        0,    None, True, True, "mg/dL",   None                                         ),
    ("Potassium",        0,    None, True, True, "mmol/L",  None                                         ),
    ("Bilirubin_total",  0,    None, True, True, "mg/dL",   "Total bilirubin"                            ),
    ("TroponinI",        0,    None, True, True, "ng/mL",   "Troponin I"                                 ),
    ("Hct",              0,    100,  True, True, "%",       "Hematocrit"                                 ),
    ("Hgb",              0,    None, True, True, "g/dL",    "Hemoglobin"                                 ),
    ("PTT",              0,    None, True, True, "seconds", "partial thromboplastin time"                ),
    ("WBC",              0,    None, True, True, "10³/µL",  "Leukocyte count"                            ),
    ("Fibrinogen",       0,    None, True, True, "mg/dL",   None                                         ),
    ("Platelets",        0,    None, True, True, "10³/µL",  "Platelet count"                             ),
    # Outcome (column 41)
    (
        "SepsisLabel",      None, None, True, True, "bool",
        ("For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6"
        " and 0 if t<tsepsis−6. For non-sepsis patients, SepsisLabel is 0.")
    ),
]  # fmt: skip
TIMESERIES_METADATA_SCHEMA = {
    "variable"        : pl.String,
    "lower_bound"     : pl.Float32,
    "upper_bound"     : pl.Float32,
    "lower_inclusive" : pl.Boolean,
    "upper_inclusive" : pl.Boolean,
    "unit"            : pl.String,
    "description"     : pl.String,
}  # fmt: skip
STATIC_COVARIATES_METADATA = [
    ("patient"    , None, None,  None, None  , None   , "Patient identifier"                        ),
    # Demographics (columns 35-40)
    ("Age"        ,    0,  100, False, True  , "years", "Years (100 for patients 90 or above)"      ),
    ("Gender"     , None, None, True , True  , "bool" , "Female (0) or Male (1)"                    ),
    ("Unit1"      ,    0,    1, True , True  , "MICU" , "Administrative identifier for ICU unit"    ),
    ("Unit2"      ,    0,    1, True , True  , "SICU" , "Administrative identifier for ICU unit"    ),
    ("HospAdmTime", None, None, True , False , "h"    , "Hours between hospital admit and ICU admit"),
]  # fmt: skip
STATIC_COVARIATES_METADATA_SCHEMA = {
    "variable"        : pl.String,
    "lower_bound"     : pl.Float32,
    "upper_bound"     : pl.Float32,
    "lower_inclusive" : pl.Boolean,
    "upper_inclusive" : pl.Boolean,
    "unit"            : pl.String,
    "description"     : pl.String,
}  # fmt: skip
TIMESERIES_SCHEMA = {
    "patient": pl.Int64,
    "time": pl.Duration(time_unit="ms"),
    **{
        name: pl.Boolean if name == "SepsisLabel" else pl.Float32
        for name, *_ in TIMESERIES_METADATA
        if name not in {"patient", "time"}
    },
}
STATIC_COVARIATES_SCHEMA = {
    "patient"     : pl.Int64,
    "Age"         : pl.Float32,
    "Gender"      : pl.String,
    "Unit1"       : pl.Boolean,
    "Unit2"       : pl.Boolean,
    "HospAdmTime" : pl.Duration(time_unit="ms"),
}  # fmt: skip
RAWDATA_SCHEMA = {
    **{
        name: pl.Float32
        for name, *_ in TIMESERIES_METADATA
        if name not in {"patient", "time", "SepsisLabel"}
    },
    "Age"         : pl.Float32,
    "Gender"      : pl.Int8,
    "Unit1"       : pl.Int8,
    "Unit2"       : pl.Int8,
    "HospAdmTime" : pl.Float32,
    "ICULOS"      : pl.Int32,
    "SepsisLabel" : pl.Int8,
}  # fmt: skip

type Key = Literal[
    "timeseries",
    "timeseries_metadata",
    "static_covariates",
    "static_covariates_metadata",
    "raw_timeseries",
    "raw_metadata",
]


class PhysioNet2019(DatasetBase[Key, pl.DataFrame]):
    r"""Physionet Challenge 2019.

    Each training data file provides a table with measurements over time. Each column of the table
    provides a sequence of measurements over time (e.g., heart rate over several hours), where the
    header of the column describes the measurement. Each row of the table provides a collection of
    measurements at the same time (e.g., heart rate and oxygen level at the same time).

    The table is formatted in the following way:

    +-----+-------+------+-----+-------------+--------+-------------+
    | HR  | O2Sat | Temp | ... | HospAdmTime | ICULOS | SepsisLabel |
    +=====+=======+======+=====+=============+========+=============+
    | NaN | NaN   | NaN  | ... | -50         | 1      | 0           |
    +-----+-------+------+-----+-------------+--------+-------------+
    | 86  | 98    | NaN  | ... | -50         | 2      | 0           |
    +-----+-------+------+-----+-------------+--------+-------------+
    | 75  | NaN   | NaN  | ... | -50         | 3      | 1           |
    +-----+-------+------+-----+-------------+--------+-------------+
    | 99  | 100   | 35.5 | ... | -50         | 4      | 1           |
    +-----+-------+------+-----+-------------+--------+-------------+

    There are 40 time-dependent variables HR, O2Sat, Temp, …, HospAdmTime which are described here.
    The final column, SepsisLabel, indicates the onset of sepsis according to the Sepsis-3
    definition, where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number)
    indicate that there was no recorded measurement of a variable at the time interval.
    """

    SOURCE_URL = r"https://physionet.org/files/challenge-2019/1.0.0/training/"
    r"""HTTP address from where the dataset can be downloaded"""
    INFO_URL = r"https://physionet.org/content/challenge-2019/1.0.0/"
    r"""HTTP address containing additional information about the dataset"""

    rawdata_files = ["training_setA.zip", "training_setB.zip"]
    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "timeseries_metadata",
        "static_covariates",
        "static_covariates_metadata",
        "raw_timeseries",
        "raw_metadata",
    ]

    rawdata_hashes = {
        "training_setA.zip": None,
        "training_setB.zip": None,
    }
    rawdata_content_hashes = {
        "training_setA.zip": "zip:C8B68113750A7A817150C723D8438BE4431716A5BADA88899A9F50D43E76DD81",
        "training_setB.zip": "zip:37ED047C7F5A0834F90BB3C03E9F17199F98489D09CFEF51CD92B61E60B960C3",
    }
    table_schemas = {
        "timeseries": TIMESERIES_SCHEMA,
        "timeseries_metadata": TIMESERIES_METADATA_SCHEMA,
        "static_covariates": STATIC_COVARIATES_SCHEMA,
        "static_covariates_metadata": STATIC_COVARIATES_METADATA_SCHEMA,
        "raw_timeseries": TIMESERIES_SCHEMA,
        "raw_metadata": STATIC_COVARIATES_SCHEMA,
    }  # fmt: skip

    rawdata_schema = RAWDATA_SCHEMA

    def read_patient_file(
        self, archive: ZipFile, /, *, compressed_file: str
    ) -> pl.DataFrame:
        r"""Read a single patient file from the archive."""
        with archive.open(compressed_file) as file:
            return pl.read_csv(
                file,
                separator="|",
                schema=self.rawdata_schema,
            ).with_columns(
                pl.col(column).cast(pl.Boolean).alias(column)
                for column in ("Gender", "Unit1", "Unit2", "SepsisLabel")
            )

    def _get_frame(self, fname: str, /) -> pl.DataFrame:
        with (
            ZipFile(self.rawdata_paths[fname], "r") as archive,
            tqdm(archive.namelist()) as iter_archive,
        ):
            iter_archive.set_description(f"Loading patient data from {fname}")
            pattern = re.compile(r"^p(?P<ID>\d{6})\.psv$")
            frames: list[pl.DataFrame] = []

            for compressed_file in iter_archive:
                match = pattern.match(compressed_file)
                if not match:
                    msg = f"Unexpected file in archive: {compressed_file!r}"
                    raise ValueError(msg)
                record_id = int(match.group("ID"))
                frames.append(
                    self.read_patient_file(
                        archive, compressed_file=compressed_file
                    ).with_columns(pl.lit(record_id, dtype=pl.Int64).alias("patient"))
                )
        self.LOGGER.info("Concatenating DataFrames")
        return pl.concat(frames)

    def _clean_all_rawdatasets(self) -> None:
        table = pl.concat([self._get_frame(fname) for fname in self.rawdata_files])

        self.LOGGER.info("Creating Timeseries Table.")
        ts = (
            table.with_columns(
                (pl.col("ICULOS").cast(pl.Int64) * 3_600_000)
                .cast(pl.Duration(time_unit="ms"))
                .alias("time")
            )
            .select(
                pl.col(column).cast(dtype).alias(column)
                for column, dtype in TIMESERIES_SCHEMA.items()
            )
            .sort("patient", "time")
        )

        self.LOGGER.info("Removing outliers from timeseries.")
        ts = remove_outliers(ts, self.timeseries_metadata)

        self.LOGGER.info("Creating Static_Covariates Table.")
        self.LOGGER.info("Validating Static_Covariates is constant.")
        static_columns = list(STATIC_COVARIATES_SCHEMA)[1:]
        for column in static_columns:
            nonconstant_values = (
                table.group_by("patient")
                .agg(pl.col(column).drop_nulls().n_unique().alias("n_unique"))
                .filter(pl.col("n_unique").gt(1))
            )
            if nonconstant_values.height:
                raise ValueError(f"Column {column} is not constant for each patient.")
        md = (
            table.group_by("patient", maintain_order=True)
            .agg(
                pl.col(column).drop_nulls().first().alias(column)
                for column in static_columns
            )
            .sort("patient")
        )

        self.LOGGER.info("Removing outliers from static_covariates.")
        md = remove_outliers(md, self.static_covariates_metadata)

        self.LOGGER.info("Finalizing static_covariates table.")
        md = (
            md.with_columns(
                (pl.col("HospAdmTime") * 3_600_000)
                .cast(pl.Duration(time_unit="ms"))
                .alias("HospAdmTime"),
                pl.when(pl.col("Gender").is_null())
                .then(pl.lit(None, dtype=pl.String))
                .when(pl.col("Gender"))
                .then(pl.lit("male"))
                .otherwise(pl.lit("female"))
                .alias("Gender"),
            )
            .select(
                pl.col(column).cast(dtype).alias(column)
                for column, dtype in STATIC_COVARIATES_SCHEMA.items()
            )
            .sort("patient")
        )

        self.serialize_table(ts, self.dataset_paths["raw_timeseries"])
        self.serialize_table(md, self.dataset_paths["raw_metadata"])

    def clean_table(self, key: Key, /) -> Optional[pl.DataFrame]:
        match key:
            case "timeseries_metadata":
                return pl.DataFrame(
                    TIMESERIES_METADATA,
                    schema=TIMESERIES_METADATA_SCHEMA,
                    orient="row",
                )
            case "static_covariates_metadata":
                return pl.DataFrame(
                    STATIC_COVARIATES_METADATA,
                    schema=STATIC_COVARIATES_METADATA_SCHEMA,
                    orient="row",
                )
            case "timeseries":
                self.LOGGER.info("Removing outliers from timeseries.")
                return remove_outliers(self.raw_timeseries, self.timeseries_metadata)
            case "static_covariates":
                self.LOGGER.info("Removing outliers from static_covariates.")
                return remove_outliers(
                    self.raw_metadata, self.static_covariates_metadata
                )
            case "raw_timeseries" | "raw_metadata":
                return self._clean_all_rawdatasets()
            case _:
                raise KeyError(f"Unknown table: {key!r} not in {self.table_names}")

    def load_table(self, key: Key, /) -> pl.DataFrame:
        r"""Load a cleaned table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])

    def get_rawdata_file(self, fname: str, /) -> None:
        r"""Download a single rawdata file."""
        # Map `training_setA.zip` -> `training_setA/`
        folder = Path(fname).with_suffix("").name + "/"
        url = f"{self.SOURCE_URL}{folder}"
        path = self.rawdata_paths[fname]
        self.LOGGER.info("Downloading %s from %s", fname, url)
        remote.download_directory_to_zip(url, path, add_toplevel_dir=False)

    def validate_rawdata(
        self, key: str | None = None, *, errors: ErrorHandler.Mode = "raise"
    ) -> bool:
        r"""Validate a single rawdata file."""
        if key is None:
            return super().validate_rawdata(errors=errors)

        expected_hash = self.rawdata_content_hashes.get(key, None)
        self.LOGGER.info(f"Validating {key!r} against hash {expected_hash!s}")
        path = self.rawdata_paths[key]
        actual_hash = hash_zip_contents(path)
        return validate_hash(actual_hash, expected_hash)
