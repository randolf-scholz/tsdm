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

__all__ = [
    # Classes
    "BeijingAirQuality",
]

import logging
import os
import zipfile
from functools import cached_property
from pathlib import Path

import pandas
from pandas import DataFrame, Timestamp, concat, read_csv, read_feather

from tsdm.datasets.base import SimpleDataset

__logger__ = logging.getLogger(__name__)


class BeijingAirQuality(SimpleDataset):
    r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa

    base_url: str = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    r"""HTTP address from where the dataset can be downloaded."""

    info_url: str = (
        r"https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data"
    )
    r"""HTTP address containing additional information about the dataset."""

    dataset: DataFrame

    @cached_property
    def rawdata_files(self) -> Path:
        r"""Location where the raw data is stored."""
        return self.rawdata_dir / "PRSA2017_Data_20130301-20170228.zip"

    def _load(self):
        r"""Load the dataset from hdf-5 file."""
        df = read_feather(self.dataset_files)
        return df.set_index("time")

    def _clean(self) -> None:
        r"""Create DataFrame with all 12 stations and `pandas.DatetimeIndex`."""

        def to_time(x):
            return Timestamp(year=x[1], month=x[2], day=x[3], hour=x[4])

        data_path = self.rawdata_dir / "PRSA_Data_20130301-20170228"

        if not os.path.exists(data_path):
            with zipfile.ZipFile(self.rawdata_files, "r") as zip_ref:
                zip_ref.extractall(self.rawdata_dir)

        __logger__.info("%s: Finished extracting dataset", self.name)

        stations = []
        for csv in os.listdir(data_path):
            print(csv)
            df = DataFrame(read_csv(data_path / csv))

            # Make multiple date columns to pandas.Timestamp
            df["time"] = df.apply(to_time, axis=1)

            # Remove date columns and index
            df = df.drop(labels=["No", "year", "month", "day", "hour"], axis=1)
            stations.append(df)

        df = concat(stations, ignore_index=True)
        data_path.unlink()
        df.name = self.name
        df = df.set_index("time")

        dtypes = {
            "wd": pandas.CategoricalDtype(),
            "station": pandas.CategoricalDtype(),
        }
        for col in df.columns:
            if col not in dtypes:
                dtypes[col] = "float32"

        df = df.astype(dtypes)
        df = df.reset_index()
        df.to_feather(self.dataset_files)
