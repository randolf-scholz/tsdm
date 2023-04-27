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
"""  # pylint: disable=line-too-long # noqa: E501

__all__ = [
    # Classes
    "BeijingAirQuality",
]

from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame

from tsdm.datasets.base import SingleTableDataset


class BeijingAirQuality(SingleTableDataset):
    r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa: E501

    BASE_URL = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = (
        r"https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data"
    )
    r"""HTTP address containing additional information about the dataset."""
    dataset_hash = (
        "sha256:32a7c2bcf2fa28e8c321777f321c0ed23fc8bdfd66090b3ad9c1191fa402bc78"
    )
    table_shape = (420768, 12)
    rawdata_files = ["PRSA2017_Data_20130301-20170228.zip"]
    rawdata_hashes = {
        "PRSA2017_Data_20130301-20170228.zip": "sha256:d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
    }

    rawdata_schema = {
        # fmt: off
        "No"      : "uint16[pyarrow]",
        "year"    : "uint16[pyarrow]",
        "month"   : "uint8[pyarrow]",
        "day"     : "uint8[pyarrow]",
        "hour"    : "uint8[pyarrow]",
        "PM2.5"   : "float32[pyarrow]",
        "PM10"    : "float32[pyarrow]",
        "SO2"     : "float32[pyarrow]",
        "NO2"     : "float32[pyarrow]",
        "CO"      : "float32[pyarrow]",
        "O3"      : "float32[pyarrow]",
        "TEMP"    : "float32[pyarrow]",
        "PRES"    : "float32[pyarrow]",
        "DEWP"    : "float32[pyarrow]",
        "RAIN"    : "float32[pyarrow]",
        "wd"      : "string",  # FIXME bug in pandas prevents using pyarrrow here.
        "station" : "string",  # FIXME bug in pandas prevents using pyarrrow here.
        "WSPM"    : "float32[pyarrow]",
        # fmt: on
    }

    def clean_table(self) -> DataFrame:
        r"""Create DataFrame with all 12 stations and `pandas.DatetimeIndex`."""
        self.LOGGER.info("Loading Data.")
        file = self.rawdata_paths["PRSA2017_Data_20130301-20170228.zip"]
        with ZipFile(file) as compressed_archive:
            stations = []
            for csv_file in compressed_archive.namelist():
                if not csv_file.endswith(".csv"):
                    self.LOGGER.warning("Skipping '%s': is not a csv-file!", csv_file)
                    continue

                with compressed_archive.open(csv_file) as compressed_file:
                    df = pd.read_csv(
                        compressed_file,
                        dtype=self.rawdata_schema,
                        index_col=0,
                    )
                    df.columns = df.columns.astype("string")
                    stations.append(df)

        self.LOGGER.info("Merging Tables.")
        table = pd.concat(stations, ignore_index=True)

        self.LOGGER.info("Adding Time Data.")
        time_cols = ["year", "month", "day", "hour"]
        table = (
            table.assign(time=pd.to_datetime(table[time_cols]))
            .drop(columns=time_cols)
            .astype({"wd": "category", "station": "category"})
            .set_index(["station", "time"])
            .sort_index()
        )
        return table
