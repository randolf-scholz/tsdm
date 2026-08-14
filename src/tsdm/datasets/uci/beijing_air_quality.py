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
"""  # ruff: ignore[E501, W505]

__all__ = [
    # Constants
    "RAWDATA_SCHEMA",
    "TIMESERIES_METADATA",
    "TIMESERIES_SCHEMA",
    "TIMESERIES_METADATA_SCHEMA",
    # Classes
    "BeijingAirQuality",
]

from typing import Literal
from zipfile import ZipFile

import polars as pl

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import remove_outliers, validate_schema

RAWDATA_SCHEMA =  {
    "No"      : pl.UInt16,
    "year"    : pl.UInt16,
    "month"   : pl.UInt8,
    "day"     : pl.UInt8,
    "hour"    : pl.UInt8,
    "PM2.5"   : pl.Float32,
    "PM10"    : pl.Float32,
    "SO2"     : pl.Float32,
    "NO2"     : pl.Float32,
    "CO"      : pl.Float32,
    "O3"      : pl.Float32,
    "TEMP"    : pl.Float32,
    "PRES"    : pl.Float32,
    "DEWP"    : pl.Float32,
    "RAIN"    : pl.Float32,
    "wd"      : pl.String,
    "WSPM"    : pl.Float32,
    "station" : pl.String,
}  # fmt: skip
TIMESERIES_SCHEMA =  {
    "station": pl.String,
    "time"   : pl.Datetime(time_unit="us"),
    "PM2.5"  : pl.Float32,
    "PM10"   : pl.Float32,
    "SO2"    : pl.Float32,
    "NO2"    : pl.Float32,
    "CO"     : pl.Float32,
    "O3"     : pl.Float32,
    "TEMP"   : pl.Float32,
    "PRES"   : pl.Float32,
    "DEWP"   : pl.Float32,
    "RAIN"   : pl.Float32,
    "wd"     : pl.Categorical,
    "WSPM"   : pl.Float32,
}  # fmt: skip
TIMESERIES_METADATA_SCHEMA = {
    "variable"        : pl.String,
    "dtype"           : pl.String,
    "lower_bound"     : pl.Float64,
    "upper_bound"     : pl.Float64,
    "lower_inclusive" : pl.Boolean,
    "upper_inclusive" : pl.Boolean,
    "unit"            : pl.String,
    "description"     : pl.String,
}  # fmt: skip
TIMESERIES_METADATA = [
    ("station", "String"     , None, None, None, None,    None, "station name"         ),
    ("time"   , "Datetime"   , None, None, None, None,   "UTC", "Measurement timestamp"),
    ("PM2.5"  , "Float32"    ,    0, None, True, True, "μg/m³", "PM2.5 concentration"  ),
    ("PM10"   , "Float32"    ,    0, None, True, True, "μg/m³", "PM10 concentration"   ),
    ("SO2"    , "Float32"    ,    0, None, True, True, "μg/m³", "SO2 concentration"    ),
    ("NO2"    , "Float32"    ,    0, None, True, True, "μg/m³", "NO2 concentration"    ),
    ("CO"     , "Float32"    ,    0, None, True, True, "μg/m³", "CO concentration"     ),
    ("O3"     , "Float32"    ,    0, None, True, True, "μg/m³", "O3 concentration"     ),
    ("TEMP"   , "Float32"    , None, None, True, True,     "℃", "temperature"          ),
    ("PRES"   , "Float32"    ,    0, None, True, True,   "hPa", "pressure"             ),
    ("DEWP"   , "Float32"    , None, None, True, True,     "℃", "dew point"            ),
    ("RAIN"   , "Float32"    ,    0, None, True, True,    "mm", "precipitation"        ),
    ("wd"     , "Categorical", None, None, True, True,    None, "wind direction"       ),
    ("WSPM"   , "Float32"    ,    0, None, True, True,   "m/s", "wind speed"           ),
]  # fmt: skip

type Key = Literal["timeseries", "timeseries_metadata"]


class BeijingAirQuality(DatasetBase[Key, pl.DataFrame]):
    r"""Hourly data set considers 6 main air pollutants and 6 relevant meteorological variables at multiple sites in Beijing.

    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Multivariate, Time-Series | **Number of Instances:**  | 420768 | **Area:**               | Physical   |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Integer, Real             | **Number of Attributes:** | 18     | **Date Donated**        | 2019-09-20 |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression                | **Missing Values?**       | Yes    | **Number of Web Hits:** | 68746      |
    +--------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # ruff: ignore[E501, W505]

    SOURCE_URL = r"https://archive.ics.uci.edu/static/public/501/"
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = (
        r"https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
    )
    r"""HTTP address containing additional information about the dataset."""

    table_names = ["timeseries", "timeseries_metadata"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["beijing+multi+site+air+quality+data.zip"]

    rawdata_hashes = {
        "beijing+multi+site+air+quality+data.zip":
            "sha256:b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4",
    }  # fmt: skip
    rawdata_schemas = {"timeseries": RAWDATA_SCHEMA}

    table_schemas = {
        "timeseries": TIMESERIES_SCHEMA,
        "timeseries_metadata": TIMESERIES_METADATA_SCHEMA,
    }
    table_shapes = {
        "timeseries": (420_768, 14),
        "timeseries_metadata": (14, 8),
    }

    def clean_timeseries(self) -> pl.DataFrame:
        rawdata_path = self.rawdata_paths["beijing+multi+site+air+quality+data.zip"]
        archive_path = "PRSA2017_Data_20130301-20170228.zip"
        rawdata_schema = self.rawdata_schemas["timeseries"]
        target_schema = self.table_schemas["timeseries"]

        with (
            ZipFile(rawdata_path) as outer_archive,
            outer_archive.open(archive_path) as inner_archive,
            ZipFile(inner_archive) as compressed_archive,
        ):
            stations: list[pl.DataFrame] = []
            for csv_file in compressed_archive.namelist():
                if not csv_file.endswith(".csv"):
                    self.LOGGER.warning("\nSkipping '%s': is not a csv-file!", csv_file)
                    continue

                with compressed_archive.open(csv_file) as compressed_file:
                    validate_schema(compressed_file, rawdata_schema)
                    stations.append(
                        pl.read_csv(
                            compressed_file,
                            schema=rawdata_schema,
                            null_values="NA",
                        ).fill_nan(None)
                    )

        self.LOGGER.info("Merging Tables.")
        table = pl.concat(stations)

        self.LOGGER.info("Adding Time Data.")
        time_cols = ["year", "month", "day", "hour"]
        ts = (
            table.drop("No")
            .with_columns(
                pl.datetime(*(pl.col(column) for column in time_cols)).alias("time")
            )
            .drop(time_cols)
            .sort("station", "time")
        )

        self.LOGGER.info("Removing outliers from timeseries.")
        ts = remove_outliers(ts, self.timeseries_metadata)

        # ensure table_schema is met
        if missing_cols := (target_schema.keys() - set(ts.columns)):
            raise ValueError(f"Missing columns in timeseries: {missing_cols}")

        ts = ts.with_columns(pl.col("wd").cast(pl.Categorical)).select(*target_schema)
        assert set(ts.get_column("wd").drop_nulls().unique().to_list()) == {
            "E", "ENE", "ESE",
            "N", "NE", "NNE", "NNW", "NW",
            "S", "SE", "SSE", "SSW", "SW",
            "W", "WNW", "WSW",
        }  # fmt: skip
        return ts

    @staticmethod
    def clean_timeseries_metadata() -> pl.DataFrame:
        r"""Create DataFrame with metadata for all 12 stations."""
        return pl.DataFrame(
            TIMESERIES_METADATA,
            schema=TIMESERIES_METADATA_SCHEMA,
            orient="row",
        )

    def load_table(self, key: Key, /) -> pl.DataFrame:
        r"""Load a cleaned dataset table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])
