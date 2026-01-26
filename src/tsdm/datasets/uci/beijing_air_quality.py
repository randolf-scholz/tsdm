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
"""  # noqa: E501, W505

__all__ = [
    # Constants
    "TIMESERIES_METADATA",
    # Classes
    "BeijingAirQuality",
]

from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame

from tsdm.datasets.base import DatasetBase
from tsdm.datasets.schemas import DEFAULT_METADATA_SCHEMA
from tsdm.datatools import InlineTable, make_dataframe, remove_outliers
from tsdm.types.aliases import TS_Keys

TIMESERIES_METADATA: InlineTable = {
    "data": [
        ("PM2.5", "float32[pyarrow]",    0, None, True, True, "μg/m³", "PM2.5 concentration"),
        ("PM10" , "float32[pyarrow]",    0, None, True, True, "μg/m³", "PM10 concentration" ),
        ("SO2"  , "float32[pyarrow]",    0, None, True, True, "μg/m³", "SO2 concentration"  ),
        ("NO2"  , "float32[pyarrow]",    0, None, True, True, "μg/m³", "NO2 concentration"  ),
        ("CO"   , "float32[pyarrow]",    0, None, True, True, "μg/m³", "CO concentration"   ),
        ("O3"   , "float32[pyarrow]",    0, None, True, True, "μg/m³", "O3 concentration"   ),
        ("TEMP" , "float32[pyarrow]", None, None, True, True, "℃"    , "temperature"        ),
        ("PRES" , "float32[pyarrow]",    0, None, True, True, "hPa"  , "pressure"           ),
        ("DEWP" , "float32[pyarrow]", None, None, True, True, "℃"    , "dew point"          ),
        ("RAIN" , "float32[pyarrow]",    0, None, True, True, "mm"   , "precipitation"      ),
        ("wd"   , "category"        , None, None, True, True, None   , "wind direction"     ),
        ("WSPM" , "float32[pyarrow]",    0, None, True, True, "m/s"  , "wind speed"         ),
    ],
    "schema": DEFAULT_METADATA_SCHEMA,
    "index": ["variable"],
}  # fmt: skip


class BeijingAirQuality(DatasetBase[TS_Keys, DataFrame]):
    r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # noqa: E501, W505

    SOURCE_URL = r"https://archive.ics.uci.edu/static/public/501/"
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = (
        r"https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
    )
    r"""HTTP address containing additional information about the dataset."""

    table_names = ["timeseries", "timeseries_metadata"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["beijing+multi+site+air+quality+data.zip"]

    rawdata_hashes = {
        "beijing+multi+site+air+quality+data.zip": \
            "sha256:b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4",
    }  # fmt: skip
    rawdata_schemas = {
        "timeseries": {
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
            "wd"      : "string[pyarrow]",
            "station" : "string[pyarrow]",
            "WSPM"    : "float32[pyarrow]",
        }
    }  # fmt: skip

    table_schemas = {
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
            "wd"    : "category",
            "WSPM"  : "float[pyarrow]",
        },
        "timeseries_metadata": DEFAULT_METADATA_SCHEMA,
    }  # fmt: skip

    def clean_timeseries(self) -> DataFrame:
        rawdata_path = self.rawdata_paths["beijing+multi+site+air+quality+data.zip"]
        archive_path = "PRSA2017_Data_20130301-20170228.zip"
        rawdata_schema = self.rawdata_schemas["timeseries"]
        target_schema = self.table_schemas["timeseries"]

        with (
            ZipFile(rawdata_path) as outer_archive,
            outer_archive.open(archive_path) as inner_archive,
            ZipFile(inner_archive) as compressed_archive,
        ):
            stations = []
            for csv_file in compressed_archive.namelist():
                if not csv_file.endswith(".csv"):
                    self.LOGGER.warning("Skipping '%s': is not a csv-file!", csv_file)
                    continue

                with compressed_archive.open(csv_file) as compressed_file:
                    df = pd.read_csv(
                        compressed_file,
                        dtype=rawdata_schema,
                        index_col=0,
                    )
                    df.columns = df.columns.astype("string[pyarrow]")
                    stations.append(df)

        self.LOGGER.info("Merging Tables.")
        table = pd.concat(stations, ignore_index=True)

        self.LOGGER.info("Adding Time Data.")
        time_cols = ["year", "month", "day", "hour"]
        ts = (
            table
            .assign(time=pd.to_datetime(table[time_cols]))
            .drop(columns=time_cols)
            .set_index(["station", "time"])
            .sort_index()
        )

        self.LOGGER.info("Removing outliers from timeseries.")
        ts = remove_outliers(ts, self.timeseries_metadata)

        self.LOGGER.info("Dropping completely missing rows.")
        ts = ts.dropna(how="all", axis="index")

        # ensure table_schema is met
        if missing_cols := (target_schema.keys() - set(ts.columns)):
            raise ValueError(f"Missing columns in static_covariates: {missing_cols}")

        ts = ts.reindex(columns=target_schema).astype(target_schema)
        assert set(ts["wd"].cat.categories) == {
            "E", "ENE", "ESE",
            "N", "NE", "NNE", "NNW", "NW",
            "S", "SE", "SSE", "SSW", "SW",
            "W", "WNW", "WSW",
        }  # fmt: skip
        return ts

    @staticmethod
    def clean_timeseries_metadata() -> DataFrame:
        r"""Create DataFrame with metadata for all 12 stations."""
        return make_dataframe(**TIMESERIES_METADATA)
