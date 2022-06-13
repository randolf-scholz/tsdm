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

import zipfile
from pathlib import Path

from pandas import Timestamp, concat, read_csv

from tsdm.datasets.base import SingleFrameDataset


class BeijingAirQuality(SingleFrameDataset):
    r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa

    BASE_URL: str = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL: str = (
        r"https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data"
    )
    r"""HTTP address containing additional information about the dataset."""

    rawdata_files = "PRSA2017_Data_20130301-20170228.zip"
    rawdata_paths: Path

    def _clean(self) -> None:
        r"""Create DataFrame with all 12 stations and `pandas.DatetimeIndex`."""

        def _to_time(x):
            return Timestamp(year=x[1], month=x[2], day=x[3], hour=x[4])

        dtypes = {
            "wd": "string",
            "station": "string",
        }

        new_dtypes = {
            "wd": "category",
            "station": "category",
        }

        self.LOGGER.info("Extracting Data.")
        with zipfile.ZipFile(self.rawdata_paths) as compressed_file:
            stations = []
            for csv_file in compressed_file.namelist():
                if not csv_file.endswith(".csv"):
                    continue
                df = read_csv(compressed_file.open(csv_file), dtype=dtypes)
                # Make multiple date columns to pandas.Timestamp
                df["time"] = df.apply(_to_time, axis=1)
                # Remove date columns and index
                df = df.drop(labels=["No", "year", "month", "day", "hour"], axis=1)
                stations.append(df)

        self.LOGGER.info("Merging Stations.")
        df = concat(stations, ignore_index=True)

        self.LOGGER.info("Setting dtypes.")
        other_columns = {
            col for col in df.columns if col not in ["time", "station", "wd"]
        }
        new_dtypes |= {col: "float32" for col in other_columns}
        df = df.astype(new_dtypes)

        self.LOGGER.info("Setting Index.")
        df = df.sort_values(by=["station", "time"])
        df = df.set_index(["station", "time"])

        return df
