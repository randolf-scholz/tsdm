r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

Beijing Multi-Site Air-Quality Data Data Set
============================================

+--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
+--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
+--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
+--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+

Source
------

Song Xi Chen, csx '@' gsm.pku.edu.cn, Guanghua School of Management, Center for Statistical Science, Peking University.

Data Set Information
--------------------

This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites.
The air-quality data are from the Beijing Municipal Environmental Monitoring Center. The meteorological data in each
air-quality site are matched with the nearest weather station from the China Meteorological Administration.
The time period is from March 1st, 2013 to February 28th, 2017. Missing data are denoted as NA.

Attribute Information
---------------------

+---------+-----------------------------------------+
| No      | row number                              |
+=========+=========================================+
| year    | year of data in this row                |
+---------+-----------------------------------------+
| month   | month of data in this row               |
+---------+-----------------------------------------+
| day     | day of data in this row                 |
+---------+-----------------------------------------+
| hour    | hour of data in this row                |
+---------+-----------------------------------------+
| PM2.5   | PM2.5 concentration (ug/m^3)            |
+---------+-----------------------------------------+
| PM10    | PM10 concentration (ug/m^3)             |
+---------+-----------------------------------------+
| SO2     | SO2 concentration (ug/m^3)              |
+---------+-----------------------------------------+
| NO2     | NO2 concentration (ug/m^3)              |
+---------+-----------------------------------------+
| CO      | CO concentration (ug/m^3)               |
+---------+-----------------------------------------+
| O3      | O3 concentration (ug/m^3)               |
+---------+-----------------------------------------+
| TEMP    | temperature (degree Celsius)            |
+---------+-----------------------------------------+
| PRES    | pressure (hPa)                          |
+---------+-----------------------------------------+
| DEWP    | dew point temperature (degree Celsius)  |
+---------+-----------------------------------------+
| RAIN    | precipitation (mm)                      |
+---------+-----------------------------------------+
| wd      | wind direction                          |
+---------+-----------------------------------------+
| WSPM    | wind speed (m/s)                        |
+---------+-----------------------------------------+
| station | name of the air-quality monitoring site |
+---------+-----------------------------------------+
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

__all__ = [
    # Classes
    "BeijingAirQuality",
]

import logging
import os
import zipfile
from pathlib import Path

import pandas
from pandas import DataFrame, Timestamp, concat, read_csv, read_hdf

from tsdm.datasets.dataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class BeijingAirQuality(BaseDataset):
    r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa

    url: str = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    info_url: str = (
        r"https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data"
    )
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def load(cls):
        """Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        df = read_hdf(cls.dataset_file, key=cls.__name__)
        df = DataFrame(df)
        return df

    @classmethod
    def clean(cls):
        r"""Create DataFrame with all 12 stations and :class:`pandas.DatetimeIndex`."""

        def to_time(x):
            return Timestamp(year=x[1], month=x[2], day=x[3], hour=x[4])

        LOGGER.info("Cleaning dataset '%s'", cls.__name__)

        file_path = cls.rawdata_path.joinpath("PRSA2017_Data_20130301-20170228.zip")
        data_path = cls.rawdata_path.joinpath("PRSA_Data_20130301-20170228")

        if not os.path.exists(data_path):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(cls.rawdata_path)

        LOGGER.info("Finished extracting dataset '%s'", cls.__name__)

        stations = []
        for csv in os.listdir(data_path):

            df = DataFrame(read_csv(data_path.joinpath(csv)))

            # Make multiple date columns to pandas.Timestamp
            df["Timestamp"] = df.apply(to_time, axis=1)
            df = df.set_index("Timestamp").reset_index()

            # Remove date columns and index
            df = df.drop(labels=["No", "year", "month", "day", "hour"], axis=1)
            stations.append(df)

        df = concat(stations, ignore_index=True)
        df.name = cls.__name__

        dtypes = {
            "wd": pandas.CategoricalDtype(),
            "station": pandas.CategoricalDtype(),
        }
        df = df.astype(dtypes)
        df.to_hdf(cls.dataset_file, key=cls.__name__, format="table")

        LOGGER.info("Finished cleaning dataset '%s'", cls.__name__)
