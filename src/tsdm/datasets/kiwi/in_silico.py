r"""In silico experiments."""

__all__ = ["InSilico"]

import shutil
from importlib import resources
from typing import Literal
from zipfile import ZipFile

import polars as pl

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import InlineTable, remove_outliers

type KEY = Literal["timeseries", "timeseries_metadata"]

_TIMESERIES_VALUE_SCHEMA = {}
TIMESERIES_METADATA: InlineTable = {
    "data": [
        ("Biomass"  , 0, None, True, True, "g/L", None),
        ("Substrate", 0, None, True, True, "g/L", None),
        ("Acetate"  , 0, None, True, True, "g/L", None),
        ("DOTm"     , 0, 100,  True, True, "%",   None),
        ("Product"  , 0, None, True, True, "g/L", None),
        ("Volume"   , 0, None, True, True, "L",   None),
        ("Feed"     , 0, None, True, True, "μL",  None),
    ],
    "schema": {
        "variable"       : pl.String,
        "lower_bound"    : pl.Float32,
        "upper_bound"    : pl.Float32,
        "lower_inclusive": pl.Boolean,
        "upper_inclusive": pl.Boolean,
        "unit"           : pl.String,
        "description"    : pl.String,
    },
}  # fmt: skip


class InSilico(DatasetBase[KEY, pl.DataFrame]):
    r"""Artificially generated data, 8 runs, 7 attributes, ~465 samples.

    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    |         | Time    | Biomass | Substrate | Acetate | DOTm  | Product | Volume    | Feed |
    +=========+=========+=========+===========+=========+=======+=========+===========+======+
    | unit    | float   | g/L     | g/l       | g/L     | %     | g/L     | L         | µL   |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    | domain  | [0, 12] | >0      | >0        | >0      | [0,1] | >0      | [0, 0.01] | >0   |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    | missing | -       | 99%     | 99%       | 99%     | 12%   | 99%     | 93%       | -    |
    +---------+---------+---------+-----------+---------+-------+---------+-----------+------+
    """

    rawdata_files = ["in_silico.zip"]
    rawdata_hashes = {
        "in_silico.zip": "sha256:ee9ad6278fb27dd933c22aecfc7b5b2501336e859a7f012cace2bb265f713cba",
    }
    table_names = ["timeseries", "timeseries_metadata"]  # pyright: ignore[reportAssignmentType]
    rawdata_schemas = {
        "timeseries": {
            "index"     : pl.Datetime(time_unit="us"),
            "Biomass"   : pl.Float32,
            "Substrate" : pl.Float32,
            "Acetate"   : pl.Float32,
            "DOTm"      : pl.Float32,
            "Product"   : pl.Float32,
            "Volume"    : pl.Float32,
            "Feed"      : pl.Float32,
        }
    }  # fmt: skip
    table_schemas = {
        "timeseries": {
            "run_id"    : pl.UInt16,
            "time"      : pl.Datetime(time_unit="us"),
            "Biomass"   : pl.Float32,
            "Substrate" : pl.Float32,
            "Acetate"   : pl.Float32,
            "DOTm"      : pl.Float32,
            "Product"   : pl.Float32,
            "Volume"    : pl.Float32,
            "Feed"      : pl.Float32,
        },
        "timeseries_metadata": TIMESERIES_METADATA["schema"],
    }  # fmt: skip
    table_shapes = {
        "timeseries": (5206, 9),
        "timeseries_metadata": (7, 7),
    }

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Create the timeseries table as a Polars DataFrame."""
        rawdata_schema = self.rawdata_schemas["timeseries"]
        with ZipFile(self.rawdata_paths["in_silico.zip"]) as files:
            runs: list[pl.DataFrame] = []
            for fname in files.namelist():
                run_id = int(fname.removesuffix(".csv"))
                with files.open(fname) as file:
                    runs.append(
                        pl.read_csv(
                            file,
                            schema=rawdata_schema,
                        )
                        .rename({"index": "time"})
                        .with_columns(pl.lit(run_id, dtype=pl.UInt16).alias("run_id"))
                    )

        ts = pl.concat(runs).sort("run_id", "time")
        cleaned_values = remove_outliers(
            ts.select(_TIMESERIES_VALUE_SCHEMA.keys()),
            self.timeseries_metadata,
            drop=False,
        )
        return ts.with_columns(cleaned_values.get_columns())

    @staticmethod
    def clean_timeseries_metadata() -> pl.DataFrame:
        r"""Create metadata for the timeseries."""
        return pl.DataFrame(
            TIMESERIES_METADATA["data"],
            schema=TIMESERIES_METADATA["schema"],
            orient="row",
        )

    def load_table(self, key: KEY, /) -> pl.DataFrame:
        r"""Load a cleaned table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])

    def get_rawdata_file(self, fname: str, /) -> None:
        r"""Download the dataset."""
        self.LOGGER.info("Copying data files into %s.", self.rawdata_paths[fname])
        if __package__ is None:
            raise ValueError(f"Unexpected package: {__package__=}")
        with resources.path(__package__, fname) as path:
            shutil.copy(path, self.rawdata_paths[fname])
