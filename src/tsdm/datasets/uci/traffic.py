r"""Daily freeway lane occupancy measurements from the San Francisco Bay Area.

The dataset contains 440 daily records with occupancy rates between 0 and 1.

| Property                       | Value                     |
|--------------------------------|---------------------------|
| **Data Set Characteristics:**  | Multivariate, Time-Series |
| **Number of Instances:**       | 440                       |
| **Area:**                      | Computer                  |
| **Attribute Characteristics:** | Real                      |
| **Number of Attributes:**      | 138672                    |
| **Date Donated**               | 2011-05-22                |
| **Associated Tasks:**          | Classification            |
| **Missing Values?**            | N/A                       |
| **Number of Web Hits:**        | 79749                     |

### Source

Source: California Department of Transportation, www.pems.dot.ca.gov
Creator: Marco Cuturi, Kyoto University, mcuturi '@' i.kyoto-u.ac.jp

### Data Set Information

The California Department of Transportation PEMS data describe freeway lane occupancy
rates between 0 and 1, sampled every ten minutes at 963 stations. Each daily record
contains 144 measurements per station. The released dataset contains 440 daily records.

The source description gives a date range of 2008-01-01 through 2009-03-30, excluding
holidays and two anomaly days. The corrected calendar used by this loader differs;
see `Traffic.clean_timeseries` for the implemented date and weekday conventions.

The classification task is to predict the day of the week for each daily record.

### Data-Format

Each split has a data file and a label file. The 440 daily records can be merged
for a different cross-validation scheme.

- `PEMS_train` contains 263 daily matrices in MATLAB syntax. For example,
  `[a b; c d]` has row vectors `[a b]` and `[c d]`. Each matrix has 963 rows
  (stations) and 144 columns (timestamps).
- `PEMS_trainlabels` gives one weekday code per training record.
- `PEMS_test` and `PEMS_testlabels` use the same format for 173 test records.
- `randperm` gives the shuffle permutation. Merge the splits and apply its inverse
  to restore chronological order.

### Attribute Information

Each attribute is an occupancy measurement at one station and timestamp.
Station identifiers are stored in `stations_list`; the PEMS website provides location
details. Each daily record contains 963 × 144 = 138,672 measurements.

### Relevant Papers

M. Cuturi, Fast Global Alignment Kernels, Proceedings of the Intern. Conference on Machine Learning 2011.
"""

__all__ = ["Traffic"]

import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Literal
from zipfile import ZipFile

import polars as pl

from tsdm.datasets.base import PolarsDataset

type Traffic_Keys = Literal["timeseries", "labels", "randperm", "invperm"]


class Traffic(PolarsDataset[Traffic_Keys]):
    r"""Daily freeway lane occupancy measurements from the San Francisco Bay Area.

    The dataset contains 440 daily records with occupancy rates between 0 and 1.

    | Property                       | Value                     |
    |--------------------------------|---------------------------|
    | **Data Set Characteristics:**  | Multivariate, Time-Series |
    | **Number of Instances:**       | 440                       |
    | **Area:**                      | Computer                  |
    | **Attribute Characteristics:** | Real                      |
    | **Number of Attributes:**      | 138672                    |
    | **Date Donated**               | 2011-05-22                |
    | **Associated Tasks:**          | Classification            |
    | **Missing Values?**            | N/A                       |
    | **Number of Web Hits:**        | 79749                     |
    """

    SOURCE_URL = r"https://archive.ics.uci.edu/static/public/204/"
    r"""URL from which the dataset can be downloaded."""
    INFO_URL = r"https://archive.ics.uci.edu/dataset/204/pems+sf"
    r"""URL containing additional information about the dataset."""

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
        self.weekdays = self._make_weekdays()
        self.dates = self._make_dates()

    def _make_weekdays(self) -> dict[int, str]:
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

    def _make_dates(self) -> pl.Series:
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

    def clean_table(self, key: Traffic_Keys, /) -> pl.DataFrame:
        r"""Create the cleaned table for the selected key."""
        match key:
            case "timeseries":
                return self.clean_timeseries()
            case "labels":
                return self.clean_labels()
            case "randperm":
                return self.clean_randperm()
            case "invperm":
                return self.clean_invperm()
            case _:
                raise KeyError(f"Key {key} unknown! Must be one of {self.table_names}")

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Create a timeseries table from the raw daily occupancy matrices.

        Each of the 440 daily records contains a 963 × 144 matrix: 963 stations
        sampled every ten minutes. The permutation tables restore calendar order.

        Notes:
            With `use_corrected_dates=True`, weekday codes run from Sunday (1)
            to Saturday (7). Dates span 2008-01-01 through 2009-03-26, excluding
            the eleven dates listed in `_make_dates`. These exclusions were
            inferred by comparing weekday labels after undoing the permutation;
            the reasons for some exclusions remain unknown.

            With `use_corrected_dates=False`, codes run from Monday (1) to
            Sunday (7). The date range ends on 2009-03-30 and excludes the
            fifteen holidays and anomaly dates listed in `_make_dates`.
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
        self.table_schemas = {**self.table_schemas, "timeseries": dict(table.schema)}
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
            self.table_schemas = {
                **self.table_schemas,
                "timeseries": dict(table.schema),
            }
        return table
