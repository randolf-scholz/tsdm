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

__all__ = [
    # Classes
    "Physionet2019",
]

import pickle
from pathlib import Path
from zipfile import ZipFile

from pandas import DataFrame, HDFStore, read_csv, read_hdf
from tqdm import tqdm

from tsdm.datasets.base import SimpleDataset


class Physionet2019(SimpleDataset):
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

    base_url: str = r"https://archive.physionet.org/users/shared/challenge-2019/"
    r"""HTTP address from where the dataset can be downloaded."""
    info_url: str = r"https://physionet.org/content/challenge-2019/1.0.0/"
    r"""HTTP address containing additional information about the dataset."""
    rawdata_files = {"A": "training_setA.zip", "B": "training_setB.zip"}
    dataset_files = "Physionet2019.h5"

    @property
    def rawdata_paths(self) -> dict[str, Path]:
        r"""Absolute paths to the raw data files."""
        return {
            key: self.rawdata_dir / path for key, path in self.rawdata_files.items()
        }

    @property
    def index(self) -> list:
        r"""Return the index of the dataset."""
        return []

    def _clean(self, store="hdf"):
        r"""Create a file representation of pandas dataframes representing the tables.

        The groups are A and B, the subgroups are the ids of the patients.

        The default is HDF5 (store='hdf'),  but in the case of this dataset it is
        much faster to pickle a dictionary of pandas data frames (store='pickle')
        In order not to change the package I override the signature in this module
        only.
        """
        dtypes = {
            # Vital signs (columns 1-8)
            "HR": None,  # Heart rate (beats per minute)
            "O2Sat": None,  # Pulse oximetry (%)
            "Temp": None,  # Temperature (Deg C)
            "SBP": None,  # Systolic BP (mm Hg)
            "MAP": None,  # Mean arterial pressure (mm Hg)
            "DBP": None,  # Diastolic BP (mm Hg)
            "Resp": None,  # Respiration rate (breaths per minute)
            "EtCO2": None,  # End tidal carbon dioxide (mm Hg)
            # Laboratory values (columns 9-34)
            "BaseExcess": None,  # Measure of excess bicarbonate (mmol/L)
            "HCO3": None,  # Bicarbonate (mmol/L)
            "FiO2": None,  # Fraction of inspired oxygen (%)
            "pH": None,  # N/A
            "PaCO2": None,  # Partial pressure of carbon dioxide from arterial blood (mm Hg)
            "SaO2": None,  # Oxygen saturation from arterial blood (%)
            "AST": None,  # Aspartate transaminase (IU/L)
            "BUN": None,  # Blood urea nitrogen (mg/dL)
            "Alkalinephos": None,  # Alkaline phosphatase (IU/L)
            "Calcium": None,  # (mg/dL)
            "Chloride": None,  # (mmol/L)
            "Creatinine": None,  # (mg/dL)
            "Bilirubin_direct": None,  # Bilirubin direct (mg/dL)
            "Glucose": None,  # Serum glucose (mg/dL)
            "Lactate": None,  # Lactic acid (mg/dL)
            "Magnesium": None,  # (mmol/dL)
            "Phosphate": None,  # (mg/dL)
            "Potassium": None,  # (mmol/L)
            "Bilirubin_total": None,  # Total bilirubin (mg/dL)
            "TroponinI": None,  # Troponin I (ng/mL)
            "Hct": None,  # Hematocrit (%)
            "Hgb": None,  # Hemoglobin (g/dL)
            "PTT": None,  # partial thromboplastin time (seconds)
            "WBC": None,  # Leukocyte count (count*10^3/µL)
            "Fibrinogen": None,  # (mg/dL)
            "Platelets": None,  # (count*10^3/µL)
            # Demographics (columns 35-40)
            "Age": None,  # Years (100 for patients 90 or above)
            "Gender": None,  # Female (0) or Male (1)
            "Unit1": None,  # Administrative identifier for ICU unit (MICU)
            "Unit2": None,  # Administrative identifier for ICU unit (SICU)
            "HospAdmTime": None,  # Hours between hospital admit and ICU admit
            "ICULOS": None,  # ICU length-of-stay (hours since ICU admit)
            # Outcome (column 41)
            # For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6
            # and 0 if t<tsepsis−6. For non-sepsis patients, SepsisLabel is 0.
            "SepsisLabel": None,
        }

        print(dtypes)

        if store == "hdf":
            h5file = HDFStore(self.dataset_files)
            for fname, prefix in [("training_setA", "A"), ("training_setB", "B")]:
                with ZipFile(self.rawdata_dir.joinpath(fname + ".zip")) as zipfile:
                    print("cleaning " + fname)
                    for zi in tqdm(zipfile.infolist()):
                        with zipfile.open(zi) as zf:
                            if zf.name.endswith("psv"):
                                df = read_csv(zf, sep="|")
                                group_name = zf.name.split(".")[-2].split("/")[1]
                                h5file.put(f"/{prefix}/{group_name}", df)
        elif store == "pickle":
            df_dict: dict[str, DataFrame] = {}
            dataset_file = self.dataset_paths.with_suffix(".pickle")
            for fname, prefix in [("training_setA", "A"), ("training_setB", "B")]:
                with ZipFile(self.rawdata_dir.joinpath(fname + ".zip")) as zipfile:
                    print("cleaning " + fname)
                    for zi in tqdm(zipfile.infolist()):
                        with zipfile.open(zi) as zf:
                            if zf.name.endswith("psv"):
                                df = read_csv(zf, sep="|")
                                group_name = zf.name.split(".")[-2].split("/")[1]
                                df_dict[f"/{prefix}/{group_name}"] = df

            with open(dataset_file, "wb") as f:
                pickle.dump(df_dict, f)
        else:
            raise Exception("store ", store, "not supported")

    def _load(self, store: str = "hdf") -> DataFrame:
        r"""Load the dataset from file.

        Default is HDF5 (store='hdf'), but in our case store='pickle' is faster.
        """
        if store == "hdf":
            with HDFStore(self.dataset_paths) as file:
                read_dfs = {}
                for root, _, files in file.walk():
                    for fn in tqdm(files):
                        key = f"{root}/{fn}"
                        read_dfs[key] = read_hdf(file, key=key)
        elif store == "pickle":
            dataset_file = self.dataset_paths.with_suffix(".pickle")
            with open(dataset_file, "rb") as f:
                read_dfs = pickle.load(f)
        else:
            raise Exception("store ", store, "not supported")

        return read_dfs
