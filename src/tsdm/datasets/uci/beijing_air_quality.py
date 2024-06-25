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
    # Constants
    "TIMESERIES_DESCRIPTION",
    # Classes
    "BeijingAirQuality",
]

from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame
from typing_extensions import Literal, TypeAlias

from tsdm.data import InlineTable, make_dataframe, remove_outliers
from tsdm.datasets.base import MultiTableDataset

KEY: TypeAlias = Literal["timeseries", "timeseries_description"]


TIMESERIES_DESCRIPTION: InlineTable = {
    "data": [
        ("PM2.5",    0, None, True, True, "μg/m³", "PM2.5 concentration"),
        ("PM10" ,    0, None, True, True, "μg/m³", "PM10 concentration" ),
        ("SO2"  ,    0, None, True, True, "μg/m³", "SO2 concentration"  ),
        ("NO2"  ,    0, None, True, True, "μg/m³", "NO2 concentration"  ),
        ("CO"   ,    0, None, True, True, "μg/m³", "CO concentration"   ),
        ("O3"   ,    0, None, True, True, "μg/m³", "O3 concentration"   ),
        ("TEMP" , None, None, True, True, "°C"   , "temperature"        ),
        ("PRES" ,    0, None, True, True, "hPa"  , "pressure"           ),
        ("DEWP" , None, None, True, True, "°C"   , "dew point"          ),
        ("RAIN" ,    0, None, True, True, "mm"   , "precipitation"      ),
        ("wd"   , None, None, True, True, None   , "wind direction"     ),
        ("WSPM" ,    0, None, True, True, "m/s"  , "wind speed"         ),
    ],
    "schema": {
        "variable"        : "string[pyarrow]",
        "lower_bound"     : "float32[pyarrow]",
        "upper_bound"     : "float32[pyarrow]",
        "lower_inclusive" : "bool[pyarrow]",
        "upper_inclusive" : "bool[pyarrow]",
        "unit"            : "string[pyarrow]",
        "description"     : "string[pyarrow]",
    },
    "index": ["variable"],
}  # fmt: skip


class BeijingAirQuality(MultiTableDataset[KEY, DataFrame]):
    r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # pylint: disable=line-too-long # noqa: E501

    SOURCE_URL = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = (
        r"https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data"
    )
    r"""HTTP address containing additional information about the dataset."""

    rawdata_files = ["PRSA2017_Data_20130301-20170228.zip"]
    rawdata_hashes = {
        "PRSA2017_Data_20130301-20170228.zip": "sha256:d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
    }

    rawdata_schema = {
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
        "wd"      : "string[pyarrow]",  # FIXME: bug in pandas prevents using pyarrow here.
        "station" : "string[pyarrow]",  # FIXME: bug in pandas prevents using pyarrow here.
        "WSPM"    : "float32[pyarrow]",
    }  # fmt: skip

    table_names = [
        "timeseries",
        "timeseries_description",
    ]  # pyright: ignore[reportAssignmentType]

    table_schemas = {  # pyright: ignore[reportAssignmentType]
        "timeseries": {
            "PM2.5" : "float[pyarrow]",
            "PM10"  : "float[pyarrow]",
            "SO2"   : "float[pyarrow]",
            "NO2"   : "float[pyarrow]",
            "CO"    : "float[pyarrow]",
            "O3"    : "float[pyarrow]",
            "TEMP"  : "float[pyarrow]",
            "PRES"  : "float[pyarrow]",
            "DEWP"  : "float[pyarrow]",
            "RAIN"  : "float[pyarrow]",
            "wd"    : "string[pyarrow]",
            "WSPM"  : "float[pyarrow]",
        },
        "timeseries_description": TIMESERIES_DESCRIPTION["schema"],
    }  # fmt: skip

    def _clean_timeseries(self) -> DataFrame:
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
        ts = (
            table.assign(time=pd.to_datetime(table[time_cols]))
            .drop(columns=time_cols)
            .set_index(["station", "time"])
            .sort_index()
        )

        self.LOGGER.info("Removing outliers from timeseries.")
        ts = remove_outliers(ts, self.timeseries_description)

        self.LOGGER.info("Dropping completely missing rows.")
        ts = ts.dropna(how="all", axis="index")

        return ts

    def clean_table(self, key: KEY) -> DataFrame:
        r"""Create DataFrame with all 12 stations and `pandas.DatetimeIndex`."""
        match key:
            case "timeseries":
                return self._clean_timeseries()
            case "timeseries_description":
                return make_dataframe(**TIMESERIES_DESCRIPTION)
            case _:
                raise KeyError(f"Unknown table: {key!r} not in {self.table_names}")
