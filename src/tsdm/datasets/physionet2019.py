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
text file. Each file will have the same header and each row will represent a single hour's worth
of data. Available patient co-variates consist of Demographics, Vital Signs, and Laboratory values,
which are defined in the tables below.

The following time points are defined for each patient:

tsuspicion

    1. Clinical suspicion of infection identified as the earlier timestamp of IV antibiotics and
       blood cultures within a specified duration.
    2. If antibiotics were given first, then the cultures must have been obtained within 24 hours.
       If cultures were obtained first, then antibiotic must have been subsequently ordered within
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

__all__ = ["Physionet2019"]

from functools import cached_property
from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame
from tqdm.autonotebook import tqdm

from tsdm.datasets.base import SingleTableDataset


class Physionet2019(SingleTableDataset):
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

    BASE_URL = r"https://archive.physionet.org/users/shared/challenge-2019/"
    r"""HTTP address from where the dataset can be downloaded"""
    INFO_URL = r"https://physionet.org/content/challenge-2019/"
    r"""HTTP address containing additional information about the dataset"""

    rawdata_files = ["training_setA.zip", "training_setB.zip"]
    rawdata_hashes = {
        "training_setA.zip": "sha256:c0def317798312e4facc0f33ac0202b3a34f412052d9096e8b122b4d3ecb7935",
        "training_setB.zip": "sha256:8a88d69a5f64bc9a87d869f527fcc2741c0712cb9a7cb1f5cdcb725336b4c8cc",
    }
    dataset_hash = (
        "sha256:1b9c868bd4c91084545ca7f159a500aa9128d07a30b6e4d47a15354029e66efe"
    )
    table_shape = (1552210, 41)

    rawdata_schema = {
        # fmt: off
        "HR"               : "float32[pyarrow]",
        "O2Sat"            : "float32[pyarrow]",
        "Temp"             : "float32[pyarrow]",
        "SBP"              : "float32[pyarrow]",
        "MAP"              : "float32[pyarrow]",
        "DBP"              : "float32[pyarrow]",
        "Resp"             : "float32[pyarrow]",
        "EtCO2"            : "float32[pyarrow]",
        "BaseExcess"       : "float32[pyarrow]",
        "HCO3"             : "float32[pyarrow]",
        "FiO2"             : "float32[pyarrow]",
        "pH"               : "float32[pyarrow]",
        "PaCO2"            : "float32[pyarrow]",
        "SaO2"             : "float32[pyarrow]",
        "AST"              : "float32[pyarrow]",
        "BUN"              : "float32[pyarrow]",
        "Alkalinephos"     : "float32[pyarrow]",
        "Calcium"          : "float32[pyarrow]",
        "Chloride"         : "float32[pyarrow]",
        "Creatinine"       : "float32[pyarrow]",
        "Bilirubin_direct" : "float32[pyarrow]",
        "Glucose"          : "float32[pyarrow]",
        "Lactate"          : "float32[pyarrow]",
        "Magnesium"        : "float32[pyarrow]",
        "Phosphate"        : "float32[pyarrow]",
        "Potassium"        : "float32[pyarrow]",
        "Bilirubin_total"  : "float32[pyarrow]",
        "TroponinI"        : "float32[pyarrow]",
        "Hct"              : "float32[pyarrow]",
        "Hgb"              : "float32[pyarrow]",
        "PTT"              : "float32[pyarrow]",
        "WBC"              : "float32[pyarrow]",
        "Fibrinogen"       : "float32[pyarrow]",
        "Platelets"        : "float32[pyarrow]",
        "Age"              : "float32[pyarrow]",
        "Gender"           : "bool[pyarrow]",
        "Unit1"            : "bool[pyarrow]",
        "Unit2"            : "bool[pyarrow]",
        "HospAdmTime"      : "float32[pyarrow]",
        "ICULOS"           : "int32[pyarrow]",
        "SepsisLabel"      : "bool[pyarrow]",
        # fmt: on
    }

    @cached_property
    def units(self) -> DataFrame:
        r"""Metadata for each unit."""
        return DataFrame(
            # fmt: off
            [
                ("HR",               (0, None), "bpm",     "Heart rate"),
                ("O2Sat",            (0, 100),  "%",       "Pulse oximetry"),
                ("Temp",             (0, None), "°C",      "Temperature"),
                ("SBP",              (0, None), "mm Hg",   "Systolic BP"),
                ("MAP",              (0, None), "mm Hg",   "Mean arterial pressure"),
                ("DBP",              (0, None), "mm Hg",   "Diastolic BP"),
                ("Resp",             (0, None), "bpm",     "Respiration rate"),
                ("EtCO2",            (0, None), "mm Hg",   "End tidal carbon dioxide"),
                # Laboratory values (columns 9-34)
                ("BaseExcess",       (None, None), "mmol/L",  "Measure of excess bicarbonate"),
                ("HCO3",             (0, None), "mmol/L",  "Bicarbonate"),
                ("FiO2",             (0, 100),  "%",       "Fraction of inspired oxygen"),
                ("pH",               (0, 14),   "pH",      "N/A"),
                ("PaCO2",            (0, None), "mm Hg",   "Partial pressure of carbon dioxide from arterial blood"),
                ("SaO2",             (0, 100),  "%",       "Oxygen saturation from arterial blood"),
                ("AST",              (0, None), "IU/L",    "Aspartate transaminase"),
                ("BUN",              (0, None), "mg/dL",   "Blood urea nitrogen"),
                ("Alkalinephos",     (0, None), "IU/L",    "Alkaline phosphatase"),
                ("Calcium",          (0, None), "mg/dL",   "N/A"),
                ("Chloride",         (0, None), "mmol/L",  "N/A"),
                ("Creatinine",       (0, None), "mg/dL",   "N/A"),
                ("Bilirubin_direct", (0, None), "mg/dL",   "Bilirubin direct"),
                ("Glucose",          (0, None), "mg/dL",   "Serum glucose"),
                ("Lactate",          (0, None), "mg/dL",   "Lactic acid"),
                ("Magnesium",        (0, None), "mmol/dL", "N/A"),
                ("Phosphate",        (0, None), "mg/dL",   "N/A"),
                ("Potassium",        (0, None), "mmol/L",  "N/A"),
                ("Bilirubin_total",  (0, None), "mg/dL",   "Total bilirubin"),
                ("TroponinI",        (0, None), "ng/mL",   "Troponin I"),
                ("Hct",              (0, 100),  "%",       "Hematocrit"),
                ("Hgb",              (0, None), "g/dL",    "Hemoglobin"),
                ("PTT",              (0, None), "seconds", "partial thromboplastin time"),
                ("WBC",              (0, None), "10^3/µL", "Leukocyte count"),
                ("Fibrinogen",       (0, None), "mg/dL",   "N/A"),
                ("Platelets",        (0, None), "10^3/µL", "N/A"),
                # Demographics (columns 35-40)
                ("Age",              (0, None), "years",   "Years (100 for patients 90 or above)"),
                ("Gender",           (0, None), "bool",    "Female (0) or Male (1)"),
                ("Unit1",            (0, None), "MICU",    "Administrative identifier for ICU unit"),
                ("Unit2",            (0, None), "SICU",    "Administrative identifier for ICU unit"),
                ("HospAdmTime",      (None, 0), "h",       "Hours between hospital admit and ICU admit"),
                ("ICULOS",           (0, None), "h",       "ICU length-of-stay (hours since ICU admit)"),
                # Outcome (column 41)
                ("SepsisLabel",      (0, None), "bool",
                    "For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6 and 0 if t<tsepsis−6."
                    " For non-sepsis patients, SepsisLabel is 0.",
                ),  # noqa: E124 closing bracket does not match visual indentation
            ],
            # fmt: on
            columns=["variable", "range", "unit", "description"],
        )

    def _get_frame(self, fname: str) -> DataFrame:
        with (
            ZipFile(self.rawdata_paths[fname], "r") as archive,
            tqdm(archive.namelist()) as iter_archive,
        ):
            iter_archive.set_description(f"Loading patient data from {fname}")

            frames = {}
            for compressed_file in iter_archive:
                if not compressed_file.endswith(".psv"):
                    continue

                record_id = compressed_file.split("/")[-1].split(".")[0]
                with archive.open(compressed_file) as file:
                    df = pd.read_csv(file, sep="|", header=0, dtype_backend="pyarrow")
                    frames[record_id] = df

        self.LOGGER.info("Concatenating DataFrames")
        table = pd.concat(frames, names=["patient", "time"]).astype(self.rawdata_schema)
        return table

    def clean_table(self) -> DataFrame:
        return pd.concat([self._get_frame(fname) for fname in self.rawdata_files])
