r"""Physionet Challenge 2019.

Physionet Challenge 2019 Data Set
=================================

The Challenge data repository contains one file per subject (e.g., training/p00101.psv for the training data).
The complete training database (42 MB), consists of two parts: training set A (20,336 subjects) and B (20,000 subjects).

Each training data file provides a table with measurements over time. Each column of the table provides a sequence
of measurements over time (e.g., heart rate over several hours), where the header of the column describes the measurement.
Each row of the table provides a collection of measurements at the same time (e.g., heart rate and oxygen level at the same time).
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
       If cultures were obtained first, then antibiotic must have been subsequently ordered within 72 hours.
    3. Antibiotics must have been administered for at least 72 consecutive hours to be considered.

tSOFA

    The occurrence of end organ damage as identified by a two-point deterioration in SOFA score within a 24-hour period.

tsepsis

    The onset time of sepsis is the earlier of tsuspicion and tSOFA as long as tSOFA occurs no more than
    24 hours before or 12 hours after tsuspicion; otherwise, the patient is not marked as a sepsis patient.
    Specifically, if tsuspicion−24≤tSOFA≤tsuspicion+12, then tsepsis=min(tsuspicion,tSOFA)

Table 1: Columns in each training data file. Vital signs (columns 1-8)
HR 	Heart rate (beats per minute)

+------------------+-----------------------------------------------------------------------------------------------------------------------+
| O2Sat            | Pulse oximetry (%)                                                                                                    |
+==================+=======================================================================================================================+
| Temp             | Temperature (Deg C)                                                                                                   |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| SBP              | Systolic BP (mm Hg)                                                                                                   |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| MAP              | Mean arterial pressure (mm Hg)                                                                                        |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| DBP              | Diastolic BP (mm Hg)                                                                                                  |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Resp             | Respiration rate (breaths per minute)                                                                                 |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| EtCO2            | End tidal carbon dioxide (mm Hg)                                                                                      |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Laboratory       | values (columns 9-34)                                                                                                 |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| BaseExcess       | Measure of excess bicarbonate (mmol/L)                                                                                |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| HCO3             | Bicarbonate (mmol/L)                                                                                                  |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| FiO2             | Fraction of inspired oxygen (%)                                                                                       |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| pH               | N/A                                                                                                                   |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| PaCO2            | Partial pressure of carbon dioxide from arterial blood (mm Hg)                                                        |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| SaO2             | Oxygen saturation from arterial blood (%)                                                                             |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| AST              | Aspartate transaminase (IU/L)                                                                                         |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| BUN              | Blood urea nitrogen (mg/dL)                                                                                           |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Alkalinephos     | Alkaline phosphatase (IU/L)                                                                                           |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Calcium          | (mg/dL)                                                                                                               |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Chloride         | (mmol/L)                                                                                                              |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Creatinine       | (mg/dL)                                                                                                               |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Bilirubin_direct | Bilirubin direct (mg/dL)                                                                                              |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Glucose          | Serum glucose (mg/dL)                                                                                                 |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Lactate          | Lactic acid (mg/dL)                                                                                                   |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Magnesium        | (mmol/dL)                                                                                                             |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Phosphate        | (mg/dL)                                                                                                               |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Potassium        | (mmol/L)                                                                                                              |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Bilirubin_total  | Total bilirubin (mg/dL)                                                                                               |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| TroponinI        | Troponin I (ng/mL)                                                                                                    |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Hct              | Hematocrit (%)                                                                                                        |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Hgb              | Hemoglobin (g/dL)                                                                                                     |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| PTT              | partial thromboplastin time (seconds)                                                                                 |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| WBC              | Leukocyte count (count*10^3/µL)                                                                                       |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Fibrinogen       | (mg/dL)                                                                                                               |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Platelets        | (count*10^3/µL)                                                                                                       |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Demographics     | (columns 35-40)                                                                                                       |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Age              | Years (100 for patients 90 or above)                                                                                  |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Gender           | Female (0) or Male (1)                                                                                                |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Unit1            | Administrative identifier for ICU unit (MICU)                                                                         |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Unit2            | Administrative identifier for ICU unit (SICU)                                                                         |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| HospAdmTime      | Hours between hospital admit and ICU admit                                                                            |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| ICULOS           | ICU length-of-stay (hours since ICU admit)                                                                            |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| Outcome          | (column 41)                                                                                                           |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
| SepsisLabel      | For sepsis patients, SepsisLabel is 1 if t≥tsepsis−6 and 0 if t<tsepsis−6. For non-sepsis patients, SepsisLabel is 0. |
+------------------+-----------------------------------------------------------------------------------------------------------------------+
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

__all__ = [
    # Classes
    "Physionet2019",
]


import logging
import pickle
from pathlib import Path
from zipfile import ZipFile

from pandas import DataFrame, HDFStore, read_csv, read_hdf
from tqdm import tqdm

from tsdm.datasets.dataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class Physionet2019(BaseDataset):
    r"""Physionet Challenge 2019.

    Each training data file provides a table with measurements over time. Each column of the table provides a sequence
    of measurements over time (e.g., heart rate over several hours), where the header of the column describes the measurement.
    Each row of the table provides a collection of measurements at the same time (e.g., heart rate and oxygen level at the same time).
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

    There are 40 time-dependent variables HR, O2Sat, Temp, ..., HospAdmTime, which are described here.
    The final column, SepsisLabel, indicates the onset of sepsis according to the Sepsis-3 definition,
    where 1 indicates sepsis and 0 indicates no sepsis. Entries of NaN (not a number) indicate that
    there was no recorded measurement of a variable at the time interval.
    """  # pylint: disable=line-too-long # noqa

    url: str = r"https://archive.physionet.org/users/shared/challenge-2019/"
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def clean(cls, store="hdf"):  # pylint: disable=arguments-differ
        r"""Create a file representation of pandas dataframes representing the tables.

        The groups are A and B, the subgroups are the ids of the patients.

        The default is HDF5 (store='hdf'),  but in the case of this dataset it is
        much faster to pickle a dictionary of pandas data frames (store='pickle')
        In order not to change the package I override the signature in this module
        only.
        """
        dataset = cls.__name__
        LOGGER.info("Cleaning dataset '%s'", dataset)
        if store == "hdf":
            # noinspection PyTypeChecker
            h5file = HDFStore(cls.dataset_file)
            for fname, prefix in [("training_setA", "A"), ("training_setB", "B")]:
                with ZipFile(cls.rawdata_path.joinpath(fname + ".zip")) as zipfile:
                    print("cleaning " + fname)
                    for zi in tqdm(zipfile.infolist()):
                        with zipfile.open(zi, "r") as zf:
                            if zf.name.endswith("psv"):
                                df = read_csv(zf, sep="|")
                                group_name = zf.name.split(".")[-2].split("/")[1]
                                h5file.put(f"/{prefix}/{group_name}", df)
        elif store == "pickle":
            dfdict = {}
            dataset_file = cls.dataset_file.with_suffix(".pickle")
            for fname, prefix in [("training_setA", "A"), ("training_setB", "B")]:
                with ZipFile(cls.rawdata_path.joinpath(fname + ".zip")) as zipfile:
                    print("cleaning " + fname)
                    for zi in tqdm(zipfile.infolist()):
                        with zipfile.open(zi, "r") as zf:
                            if zf.name.endswith("psv"):
                                df = read_csv(zf, sep="|")
                                group_name = zf.name.split(".")[-2].split("/")[1]
                                dfdict[f"/{prefix}/{group_name}"] = df

            with open(dataset_file, "wb") as f:
                pickle.dump(dfdict, f)
        else:
            raise Exception("store ", store, "not supported")

        LOGGER.info("Finished extracting dataset '%s'", dataset)

    # noinspection PyTypeChecker
    @classmethod
    def load(cls, store="hdf"):  # pylint: disable=arguments-differ
        r"""Load the dataset from file.

        Default is HDF5 (store='hdf'), but in our case store='pickle' is faster.
        """
        super().load()  # <- makes sure DS is downloaded and preprocessed
        if store == "hdf":
            with HDFStore(cls.dataset_file) as file:
                read_dfs = {}
                for root, _, files in file.walk():
                    for fn in tqdm(files):
                        key = f"{root}/{fn}"
                        read_dfs[key] = read_hdf(file, key=key)
        elif store == "pickle":
            dataset_file = cls.dataset_file.with_suffix(".pickle")
            with open(dataset_file, "rb") as f:
                read_dfs = pickle.load(f)
        else:
            raise Exception("store ", store, "not supported")

        return read_dfs
