r"""Data set contains electricity consumption of 370 points/clients.

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
"""  # noqa: E501, W505

__all__ = ["Electricity"]

from typing import Literal
from zipfile import ZipFile

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import validate_schema


class Electricity(DatasetBase[Literal["timeseries"], pl.DataFrame]):
    r"""Data set containing electricity consumption of 370 points/clients.

    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**  | Time-Series            | **Number of Instances:**  | 370    | **Area:**               | Computer   |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:** | Real                   | **Number of Attributes:** | 140256 | **Date Donated**        | 2015-03-13 |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**          | Regression, Clustering | **Missing Values?**       | N/A    | **Number of Web Hits:** | 93733      |
    +--------------------------------+------------------------+---------------------------+--------+-------------------------+------------+

    Notes:
        More than 200 channels are completly missing before 2012-01-01 00:15:00.
        There are 3 extra dates with large number of missing values:

            - 2012-03-25
            - 2013-03-31
            - 2014-03-30

        More specifically, the timestamps:

            - 2011-MM-DD hh:mm:ss
            - 2012-01-01 00:00:00
            - 2012-03-25 01:00:00
            - 2012-03-25 01:15:00
            - 2012-03-25 01:30:00
            - 2012-03-25 01:45:00
            - 2013-03-31 01:00:00
            - 2013-03-31 01:15:00
            - 2013-03-31 01:30:00
            - 2013-03-31 01:45:00
            - 2014-03-30 01:00:00
            - 2014-03-30 01:15:00
            - 2014-03-30 01:30:00
            - 2014-03-30 01:45:00

    Recommendation:
        At the given dates, replace zero with NaN.
    """  # noqa: E501, W505

    SOURCE_URL = r"https://archive.ics.uci.edu/static/public/321/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = (
        r"https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"
    )
    r"""HTTP address containing additional information about the dataset."""

    rawdata_files = ["electricityloaddiagrams20112014.zip"]
    rawdata_hashes = {
        "electricityloaddiagrams20112014.zip": \
            "sha256:f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1",
    }  # fmt: skip
    table_names = ["timeseries"]  # pyright: ignore[reportAssignmentType]
    rawdata_schemas = {
        "timeseries": {
            "": pl.Datetime(time_unit="us"),
            **{f"MT_{client:03d}": pl.Float64 for client in range(1, 371)},
        }
    }
    table_schemas = {
        "timeseries": {
            "time": pl.Datetime(time_unit="us"),
            **{f"MT_{client:03d}": pl.Float64 for client in range(1, 371)},
        }
    }
    table_shapes = {"timeseries": (140256, 371)}

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Create a Polars DataFrame with one column per client."""
        rawdata_path = self.rawdata_paths["electricityloaddiagrams20112014.zip"]
        rawdata_schema = self.rawdata_schemas["timeseries"]
        with (
            ZipFile(rawdata_path) as archive,
            archive.open("LD2011_2014.txt") as file,
        ):
            validate_schema(file, rawdata_schema, separator=";")
            ts = pl.read_csv(
                file,
                new_columns=["time"],
                separator=";",
                decimal_comma=True,
                schema=rawdata_schema,
            ).fill_nan(None)

        return ts

    def load_table(self, key: Literal["timeseries"], /) -> pl.DataFrame:
        r"""Load a cleaned dataset table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])

    def make_zero_plot(self) -> Axes:
        r"""Plot number of zero values per timestamp."""
        ts = self.timeseries
        zero_counts = ts.select(
            pl.sum_horizontal(pl.exclude("time").le(0)).alias("zero-values")
        ).to_series()
        _, axes = plt.subplots()
        axes.plot(ts.get_column("time").to_numpy(), zero_counts.to_numpy())
        axes.set_ylabel("zero-values")
        return axes
