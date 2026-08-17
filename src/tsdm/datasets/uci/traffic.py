r"""15 months worth of daily data (440 daily records) that describes the occupancy rate, between 0 and 1, of different car lanes of the San Francisco bay area freeways across time.

+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Data Set Characteristics:**   | Multivariate, Time-Series | **Number of Instances:**  | 440    | **Area:**               | Computer   |
+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Attribute Characteristics:**  | Real                      | **Number of Attributes:** | 138672 | **Date Donated**        | 2011-05-22 |
+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
| **Associated Tasks:**           | Classification            | **Missing Values?**       | N/A    | **Number of Web Hits:** | 79749      |
+---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+

Source
------
Source: California Department of Transportation, www.pems.dot.ca.gov
Creator: Marco Cuturi, Kyoto University, mcuturi '@' i.kyoto-u.ac.jp

Data Set Information
--------------------
We have downloaded 15 months worth of daily data from the California Department of Transportation PEMS website, [Web Link], The data describes the occupancy
rate, between 0 and 1, of different car lanes of San Francisco bay area freeways. The measurements cover the period from Jan. 1st 2008 to Mar. 30th 2009 and are sampled every 10 minutes. We consider each day in this database as a single time series of dimension 963 (the number of sensors which functioned consistently throughout the studied period) and length 6 x 24=144. We remove public holidays from the dataset, as well
as two days with anomalies (March 8th 2009 and March 9th 2008) where all sensors were muted between 2:00 and 3:00 AM. This results in a database of 440 time series.

The task we propose on this dataset is to classify each observed day as the correct day of the week, from Monday to Sunday, e.g. label it with an integer in {1,2,3,4,5,6,7}.

I will keep separate copies of this database on my website in a Matlab format. If you use Matlab, it might be more convenient to consider these .mat files directly.

Data-Format
-----------
There are two files for each fold, the data file and the labels file. We have key the 440 time series between train and test folds, but you are of course free to merge them to consider a different cross validation setting.
- The PEMS_train textfile has 263 lines. Each line describes a time-series provided as a matrix. The matrix syntax is that of Matlab, e.g. [ a b ; c d] is the matrix with row vectors [a b] and [c d] in that order. Each matrix describes the different occupancies rates (963 lines, one for each station/detector) sampled every 10 minutes during the day (144 columns).
- The PEMS_trainlabel text describes, for each day of measurements described above, the day of the week on which the data was sampled, namely an integer between 1 (Mon.) and 7 (Sun.).

- PEMS_test and PEMS_testlabels are formatted in the same way, except that there are 173 test instances.

- The permutation that I used to shuffle the dataset is given in the randperm file. If you need to rearrange the data so that it follows the calendar order, you should merge train and test samples and reorder them using the inverse permutation of randperm.

Attribute Information
---------------------
Each attribute describes the measurement of the occupancy rate (between 0 and 1) of a captor location as recorded by a measuring station, at a given timestamp in time during the day. The ID of each station is given in the stations_list text file. For more information on the location (GPS, Highway, Direction) of each station please refer to the PEMS website. There are 963 (stations) x 144 (timestamps) = 138.672 attributes for each record.

Relevant Papers
---------------
M. Cuturi, Fast Global Alignment Kernels, Proceedings of the Intern. Conference on Machine Learning 2011.
"""  # ruff: ignore[E501, W505]

__all__ = ["Traffic"]

import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from functools import cached_property
from io import StringIO
from typing import Any, Literal
from zipfile import ZipFile

import polars as pl

from tsdm.datasets.base import PolarsDataset

type Traffic_Keys = Literal["timeseries", "labels", "randperm", "invperm"]


class Traffic(PolarsDataset[Traffic_Keys]):
    r"""15 months worth of daily data (440 daily records) that describes the occupancy rate, between 0 and 1, of different car lanes of the San Francisco bay area freeways across time.

    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Data Set Characteristics:**   | Multivariate, Time-Series | **Number of Instances:**  | 440    | **Area:**               | Computer   |
    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Attribute Characteristics:**  | Real                      | **Number of Attributes:** | 138672 | **Date Donated**        | 2011-05-22 |
    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    | **Associated Tasks:**           | Classification            | **Missing Values?**       | N/A    | **Number of Web Hits:** | 79749      |
    +---------------------------------+---------------------------+---------------------------+--------+-------------------------+------------+
    """  # ruff: ignore[E501, W505]

    SOURCE_URL = r"https://archive.ics.uci.edu/static/public/204/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL = r"https://archive.ics.uci.edu/dataset/204/pems+sf"
    r"""HTTP address containing additional information about the dataset."""

    rawdata_files = ["pems+sf.zip"]
    rawdata_hashes = {
        "pems+sf.zip": (
            "sha256:371d15048b5401026396d4587e5f9be79792e06d74f7a42a0ec84975e692147e"
        )
    }

    table_names = [  # pyright: ignore[reportAssignmentType]
        "timeseries",
        "labels",
        "randperm",
        "invperm",
    ]
    table_schemas = {  # pyright: ignore[reportAssignmentType]
        "timeseries": defaultdict(lambda: pl.Float32, {"time": pl.Duration("us")}),
        "labels": {
            "day": pl.Datetime(time_unit="us"),
            "label": pl.UInt8,
        },
        "randperm": {"randperm": pl.UInt16},
        "invperm": {"invperm": pl.UInt16},
    }
    table_shapes = {  # pyright: ignore[reportAssignmentType]
        "timeseries": (63_360, 964),
        "labels": (440, 2),
        "randperm": (440, 1),
        "invperm": (440, 1),
    }

    timeseries: pl.DataFrame
    labels: pl.DataFrame
    randperm: pl.DataFrame
    invperm: pl.DataFrame

    def __init__(self, *, use_corrected_dates: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_corrected_dates = use_corrected_dates

    @cached_property
    def weekdays(self) -> dict[int, str]:
        r"""Encoding of weekdays."""
        if self.use_corrected_dates:
            weekdays = {
                1: "Sunday",
                2: "Monday",
                3: "Tuesday",
                4: "Wednesday",
                5: "Thursday",
                6: "Friday",
                7: "Saturday",
            }
        else:
            weekdays = {
                1: "Monday",
                2: "Tuesday",
                3: "Wednesday",
                4: "Thursday",
                5: "Friday",
                6: "Saturday",
                7: "Sunday",
            }
        return weekdays

    @cached_property
    def dates(self) -> pl.Series:
        r"""Dates of the dataset."""
        match self.use_corrected_dates:
            case True:
                end = datetime(2009, 3, 26)
                anomalies = (
                    datetime(2008, 1, 1),  # New Year’s Day
                    datetime(2008, 1, 21),  # Martin Luther King Jr. Day
                    datetime(2008, 2, 18),  # Washington’s Birthday
                    datetime(2008, 3, 9),  # anomaly
                    datetime(2008, 5, 26),  # Memorial Day
                    datetime(2008, 7, 4),  # Independence Day
                    datetime(2008, 9, 1),  # Labor Day
                    datetime(2008, 10, 20),  # ???
                    datetime(2008, 11, 17),  # ???
                    datetime(2008, 12, 7),  # ???
                    datetime(2009, 2, 23),  # ???
                )
            case False:
                end = datetime(2009, 3, 30)
                anomalies = (
                    datetime(2008, 1, 1),    # New Year’s Day
                    datetime(2008, 1, 21),   # Martin Luther King Jr. Day
                    datetime(2008, 2, 18),   # Washington’s Birthday
                    datetime(2008, 3, 9),    # Anomaly day
                    datetime(2008, 5, 26),   # Memorial Day
                    datetime(2008, 7, 4),    # Independence Day
                    datetime(2008, 9, 1),    # Labor Day
                    datetime(2008, 10, 13),  # Columbus Day
                    datetime(2008, 11, 11),  # Veterans Day
                    datetime(2008, 11, 27),  # Thanksgiving
                    datetime(2008, 12, 25),  # Christmas Day
                    datetime(2009, 1, 1),    # New Year’s Day
                    datetime(2009, 1, 19),   # Martin Luther King Jr. Day
                    datetime(2009, 2, 16),   # Washington’s Birthday
                    datetime(2009, 3, 8),    # Anomaly day
                )  # fmt: skip

        dates = pl.datetime_range(
            datetime(2008, 1, 1),
            end,
            interval="1d",
            eager=True,
        )
        return dates.filter(~dates.is_in(anomalies)).alias("day")

    @staticmethod
    def _parse_timeseries(
        content: str, /, *, day: datetime, stations: Sequence[str]
    ) -> pl.DataFrame:
        r"""Parse one daily 963 × 144 occupancy matrix."""
        time_columns = [f"time_{index}" for index in range(144)]
        matrix = pl.read_csv(
            StringIO(
                content.replace("[", "")
                .replace("]", "")
                .replace(";", "\n")
                .replace(" ", ",")
            ),
            has_header=False,
            new_columns=time_columns,
            schema=dict.fromkeys(time_columns, pl.Float32),
        ).transpose(column_names=stations)
        timestamps = pl.datetime_range(
            day,
            day + timedelta(hours=23, minutes=50),
            interval="10m",
            eager=True,
        ).alias("time")
        return matrix.with_columns(timestamps).select(
            pl.col("time"), pl.exclude("time")
        )

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Create DataFrame from raw data.

        Notes:
            Sampling rate = 10 minutes => 144 samples/day

        Despite the description of the dataset stating:

        The task we propose on this dataset is to classify each observed day as the correct day of the week,
        from Monday to Sunday, e.g. label it with an integer in {1,2,3,4,5,6,7}.

        In truth, '1' encodes as Sunday and '7' as Saturday.

        PEMS_train: 267 rows

        - each row is data for 1 day.
        - each row encodes a 963×144 matrix (stations×timestamps)

        PEMS_test: same but only 172 rows
        station_labels: labels of the 963 stations

        In total, 440 days of observations.

        - original data range is 455 days: 2008-01-01 - 2009-03-30 (15 months)
        - authors manually removed holidays as well as 2 anomalies: 2008-03-09 and 2009-03-08.
        - in total 10 days missing.

        The authors of N-BEATS guesstimate the missing days to be:

        1. Jan. 1, 2008 (New Year’s Day)
        2. Jan. 21, 2008 (Martin Luther King Jr. Day)
        3. Feb. 18, 2008 (Washington’s Birthday)
        4. Mar. 9, 2008 (Anomaly day)
        5. May 26, 2008 (Memorial Day)
        6. Jul. 4, 2008 (Independence Day)
        7. Sep. 1, 2008 (Labor Day)
        8. Oct. 13, 2008 (Columbus Day)
        9. Nov. 11, 2008 (Veterans Day)
        10. Nov. 27, 2008 (Thanksgiving)
        11. Dec. 25, 2008 (Christmas Day)
        12. Jan. 1, 2009 (New Year’s Day)
        13. Jan. 19, 2009 (Martin Luther King Jr. Day)
        14. Feb. 16, 2009 (Washington’s Birthday)
        15. Mar. 8, 2009 (Anomaly day)

        However, there was a big mistake made: they assumed the value `1` encodes
        Monday, when it actually encodes Sunday.

        The true missing dates appear to be by reverse-engineering:

        - "2008-01-02": "1 day off New Year’s Day",
        - "2008-01-22": "1 day off Martin Luther King Jr. Day",
        - "2008-02-19": "1 day off Washington’s Birthday",
        - "2008-03-10": "1 day off anomaly + wrong year",
        - "2008-05-27": "1 day off Memorial Day",
        - "2008-07-05": "1 day off Independence Day",
        - "2008-09-02": "1 day off Labor Day",
        - "2008-10-21": "???",
        - "2008-11-18": "???",
        - "2008-12-08": "???",
        - "2009-02-24": "???",

        The true anomalies were found by iteratively adding days one by one,
        Each time checking when the first date was when `labels[invperm].map(weekdays)`
        didn't match with `dates.day_name()`
        """
        randperm = self.randperm.to_series(0)
        dates = self.dates.gather(randperm)
        days = iter(dates.to_list())

        with ZipFile(self.rawdata_paths["pems+sf.zip"]) as archive:
            stations = (
                archive.read("stations_list").decode("utf8").strip("[]\n").split()
            )
            if len(stations) != 963:
                raise ValueError("Expected 963 station identifiers in raw data.")

            frames: list[pl.DataFrame] = []
            for filename in ("PEMS_train", "PEMS_test"):
                with archive.open(filename) as file:
                    frames.extend(
                        self._parse_timeseries(
                            line.decode("utf8"),
                            day=next(days),
                            stations=stations,
                        )
                        for line in file
                    )

        if next(days, None) is not None:
            raise ValueError(
                "Number of raw time series does not match number of dates."
            )
        table = pl.concat(frames, rechunk=True)
        self.table_schemas = self.table_schemas | {"timeseries": dict(table.schema)}  # type: ignore
        return table

    @staticmethod
    def _parse_labels(content: str, /) -> pl.Series:
        r"""Parse labels from a raw PEMS label file."""
        return pl.Series(
            "label",
            [int(value) for value in content.replace("[", "").replace("]", "").split()],
            dtype=pl.UInt8,
        )

    def clean_labels(self) -> pl.DataFrame:
        r"""Clean the PEMS-SF weekday labels."""
        randperm = self.randperm.to_series(0)
        invperm = self.invperm.to_series(0)
        shuffled_dates = self.dates.gather(randperm)

        with ZipFile(self.rawdata_paths["pems+sf.zip"]) as archive:
            trainlabels = self._parse_labels(
                archive.read("PEMS_trainlabels").decode("utf8")
            )
            testlabels = self._parse_labels(
                archive.read("PEMS_testlabels").decode("utf8")
            )

        train_dates = shuffled_dates.slice(0, len(trainlabels))
        test_dates = shuffled_dates.slice(len(trainlabels))
        for labels, dates in (
            (trainlabels, train_dates),
            (testlabels, test_dates),
        ):
            expected_weekdays = dates.dt.strftime("%A").to_list()
            actual_weekdays = [self.weekdays[int(label)] for label in labels.to_list()]
            if actual_weekdays != expected_weekdays:
                raise ValueError("Labels do not match with dates!")

        labels = pl.concat(
            [
                pl.DataFrame({"day": train_dates, "label": trainlabels}),
                pl.DataFrame({"day": test_dates, "label": testlabels}),
            ],
            rechunk=True,
        )

        unshuffled_labels = labels.get_column("label").gather(invperm)
        matches = [
            self.weekdays[int(label)] == day.strftime("%A")
            for label, day in zip(
                unshuffled_labels.to_list(),
                self.dates.to_list(),
                strict=True,
            )
        ]
        if all(matches):
            self.LOGGER.info("All encoded labels match with the day name!")
        else:
            warnings.warn(
                f"Mismatches detected for {len(matches) - sum(matches)} labels.",
                UserWarning,
                stacklevel=2,
            )

        return labels

    def clean_randperm(self) -> pl.DataFrame:
        r"""Create the zero-indexed permutation table."""
        with (
            ZipFile(self.rawdata_paths["pems+sf.zip"]) as archive,
            archive.open("randperm") as file,
        ):
            values = [
                int(value) - 1
                for value in (
                    file.read().decode("utf8").replace("[", "").replace("]", "").split()
                )
            ]
        return pl.Series("randperm", values, dtype=pl.UInt16).to_frame()

    def clean_invperm(self) -> pl.DataFrame:
        r"""Create the inverse-permutation table."""
        randperm = self.randperm.to_series(0)
        invperm = randperm.arg_sort().cast(pl.UInt16).alias("invperm")
        expected = pl.Series("randperm", range(len(randperm)), dtype=pl.UInt16)
        if not randperm.gather(invperm).equals(expected):
            raise ValueError("Inverse permutation does not match!")
        return invperm.to_frame()

    def load_table(self, key: Traffic_Keys, /) -> pl.DataFrame:
        r"""Load a cleaned table as a Polars DataFrame."""
        table = pl.read_parquet(self.dataset_paths[key])
        if key == "timeseries":
            self.table_schemas = self.table_schemas | {"timeseries": dict(table.schema)}  # type: ignore
        return table
