r"""Electricity consumption measurements for 370 clients.

### ElectricityLoadDiagrams20112014 Data Set

| Property                       | Value                  |
|--------------------------------|------------------------|
| **Data Set Characteristics:**  | Time-Series            |
| **Number of Instances:**       | 370                    |
| **Area:**                      | Computer               |
| **Attribute Characteristics:** | Real                   |
| **Number of Attributes:**      | 140256                 |
| **Date Donated**               | 2015-03-13             |
| **Associated Tasks:**          | Regression, Clustering |
| **Missing Values?**            | N/A                    |
| **Number of Web Hits:**        | 93733                  |

### Source

Artur Trindade, artur.trindade '@' elergone.pt, Elergone, NORTE-07-0202-FEDER-038564
Data type: TS
Task: regression, clustering
Number of Instances (records in your data set):370
Number of Attributes (fields within each record):140256

### Data Set Information

The source encodes missing consumption as zero rather than as explicit nulls.
Measurements are in kW at 15-minute intervals; divide by four to obtain kWh.
Each column represents one client. Clients added after 2011 have zero values
before their first measurements.

Timestamps use Portuguese local time, with 96 measurements per day. On the March
daylight saving transition, values between 01:00 and 02:00 are zero. On the October
transition, values in that interval aggregate consumption from two hours.

### Attribute Information

The source is a semicolon-delimited text file. The first column contains timestamps
in `yyyy-mm-dd hh:mm:ss` format; the remaining columns contain consumption in kW.
"""

__all__ = ["Electricity"]

from typing import Literal
from zipfile import ZipFile

import polars as pl

from tsdm.datasets.base import PolarsDataset
from tsdm.validation import validate_csv_header


class Electricity(PolarsDataset[Literal["timeseries"]]):
    r"""Electricity consumption measurements for 370 clients.

    | Property                       | Value                  |
    |--------------------------------|------------------------|
    | **Data Set Characteristics:**  | Time-Series            |
    | **Number of Instances:**       | 370                    |
    | **Area:**                      | Computer               |
    | **Attribute Characteristics:** | Real                   |
    | **Number of Attributes:**      | 140256                 |
    | **Date Donated**               | 2015-03-13             |
    | **Associated Tasks:**          | Regression, Clustering |
    | **Missing Values?**            | N/A                    |
    | **Number of Web Hits:**        | 93733                  |

    Notes:
        The source encodes missing measurements as zeros. Plotting the number
        of zero-valued clients at each timestamp reveals two patterns:

        - An elevated zero count before 2012: 210 channels are entirely zero
          throughout 2011 and at 2012-01-01 00:00. The count drops from 210
          to 50 at 2012-01-01 00:15 as additional clients begin reporting.
        - Three sharp spikes at the spring daylight saving transitions:

            - 2012-03-25: up to 251 zero-valued channels.
            - 2013-03-31: up to 242 zero-valued channels.
            - 2014-03-30: up to 253 zero-valued channels.

          Each spike spans the four readings at 01:00, 01:15, 01:30, and 01:45
          in Portuguese local time. Zeros representing these gaps should be
          treated as missing measurements rather than measured consumption.

        The following example makes these patterns visible. It requires
        downloading and preprocessing the dataset and is intended for
        interactive use.

        >>> from matplotlib import pyplot as plt  # doctest: +SKIP
        ...
        ... from tsdm.datasets import Electricity
        ... from tsdm.plotting import plot_zero_count
        ...
        ... dataset = Electricity()
        ... ts = dataset.timeseries
        ... ax = plot_zero_count(
        ...     ts.drop("time").to_numpy(),
        ...     time=ts.get_column("time").to_numpy(),
        ... )
        ... # The elevated band in 2011 reflects the 210 missing channels.
        ... # The three spikes mark 2012-03-25, 2013-03-31, and 2014-03-30.
        ... plt.show()
    """

    SOURCE_URL = r"https://archive.ics.uci.edu/static/public/321/"
    r"""URL from which the dataset can be downloaded."""
    INFO_URL = (
        r"https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"
    )
    r"""URL containing additional information about the dataset."""

    rawdata_files = ["electricityloaddiagrams20112014.zip"]
    rawdata_hashes = {
        "electricityloaddiagrams20112014.zip":
            "sha256:f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1",
    }  # fmt: skip
    rawdata_schemas = {
        "timeseries": {
            "": pl.Datetime(time_unit="us"),
            **{f"MT_{client:03d}": pl.Float32 for client in range(1, 371)},
        }
    }

    table_names = ["timeseries"]  # pyright: ignore[reportAssignmentType]
    table_schemas = {  # pyright: ignore[reportAssignmentType]
        "timeseries": {
            "time": pl.Datetime(time_unit="us"),
            **{f"MT_{client:03d}": pl.Float32 for client in range(1, 371)},
        }
    }
    table_shapes = {"timeseries": (140_256, 371)}  # pyright: ignore[reportAssignmentType]

    def clean_table(self, key: Literal["timeseries"], /) -> pl.DataFrame:
        r"""Create the cleaned timeseries table."""
        match key:
            case "timeseries":
                return self.clean_timeseries()
            case _:
                raise KeyError(f"Key {key} unknown! Must be one of {self.table_names}")

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Create a Polars DataFrame with one column per client."""
        rawdata_path = self.rawdata_paths["electricityloaddiagrams20112014.zip"]
        rawdata_schema = self.rawdata_schemas["timeseries"]
        with (
            ZipFile(rawdata_path) as archive,
            archive.open("LD2011_2014.txt") as file,
        ):
            validate_csv_header(file, rawdata_schema, separator=";")
            ts = (
                pl.read_csv(
                    file,
                    separator=";",
                    decimal_comma=True,
                    schema=rawdata_schema,
                )
                .fill_nan(None)
                .rename({"": "time"})
            )

        return ts
