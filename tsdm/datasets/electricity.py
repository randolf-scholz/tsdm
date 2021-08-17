r"""Data set containing electricity consumption of 370 points/clients.

ElectricityLoadDiagrams20112014 Data Set
========================================

+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
| **Data Set Characteristics:**  | Time-Series            | **Number of Instances:**  | 370    | **Area:**               | Computer   |
+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
| **Attribute Characteristics:** | Real                   | **Number of Attributes:** | 140256 | **Date Donated**        | 2015-03-13 |
+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
| **Associated Tasks:**          | Regression, Clustering | **Missing Values?**       | N/A    | **Number of Web Hits:** | 93733      |
+--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+

Source
------

Artur Trindade, artur.trindade '@' elergone.pt, Elergone, NORTE-07-0202-FEDER-038564
Data type: TS
Task: regression, clustering
Number of Instances (records in your data set):370
Number of Attributes (fields within each record):140256

Data Set Information
--------------------

Data set has no missing values.
Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.
Each column represent one client. Some clients were created after 2011. In these cases consumption were considered zero.
All time labels report to Portuguese hour. However all days present 96 measures (24*4). Every year in March time change
day (which has only 23 hours) the values between 1:00 am and 2:00 am are zero for all points. Every year in October time
change day (which has 25 hours) the values between 1:00 am and 2:00 am aggregate the consumption of two hours.

Attribute Information
---------------------

Data set were saved as txt using csv format, using semi colon (;).
First column present date and time as a string with the following format 'yyyy-mm-dd hh:mm:ss'
Other columns present float values with consumption in kW
"""  # pylint: disable=line-too-long # noqa

import logging
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import numpy as np
from pandas import DataFrame, read_csv, read_hdf

from tsdm.datasets.dataset import BaseDataset

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["Electricity"]


class Electricity(BaseDataset):
    r"""Data set containing electricity consumption of 370 points/clients.

    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Time-Series            | **Number of Instances:**  | 370    | **Area:**               | Computer   |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Real                   | **Number of Attributes:** | 140256 | **Date Donated**        | 2015-03-13 |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression, Clustering | **Missing Values?**       | N/A    | **Number of Web Hits:** | 93733      |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa

    url: str = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00321/"
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def clean(cls):
        r"""Create DataFrame with 1 column per client and :class:`pandas.DatetimeIndex`."""
        dataset = cls.__name__
        logger.info("Cleaning dataset '%s'", dataset)

        fname = "LD2011_2014.txt"
        with ZipFile(cls.rawdata_path.joinpath(fname + ".zip")) as files:
            files.extract(fname, path=cls.dataset_path)

        logger.info("Finished extracting dataset '%s'", dataset)

        df = read_csv(
            cls.dataset_path.joinpath(fname),
            sep=";",
            decimal=",",
            parse_dates=[0],
            index_col=0,
            dtype=np.float64,
        )

        df = df.rename_axis(index="time", columns="client")
        df.name = f"{dataset}"
        df.to_hdf(cls.dataset_file, key=f"{dataset}")
        cls.dataset_path.joinpath(fname).unlink()

        logger.info("Finished cleaning dataset '%s'", dataset)

    @classmethod
    def load(cls):
        """Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        df = read_hdf(cls.dataset_file, key=cls.__name__)
        df = DataFrame(df)
        return df
