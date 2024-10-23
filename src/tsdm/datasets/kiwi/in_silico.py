r"""In silico experiments."""

__all__ = ["InSilico"]

import shutil
from importlib import resources
from typing import Literal
from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame

from tsdm.data import InlineTable, make_dataframe, remove_outliers
from tsdm.datasets.base import DatasetBase

type KEY = Literal["timeseries", "timeseries_metadata"]


class InSilico(DatasetBase[KEY, DataFrame]):
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
    table_shapes = {"timeseries": (5206, 7)}

    def clean_timeseries(self) -> DataFrame:
        with ZipFile(self.rawdata_paths["in_silico.zip"]) as files:
            dfs = {}
            for fname in files.namelist():
                key = int(fname.split(".csv")[0])
                with files.open(fname) as file:
                    df = pd.read_csv(file, index_col=0, parse_dates=[0], dayfirst=True)
                    dfs[key] = df.rename_axis(index="time")

        # Set index, dtype and sort.
        ts = (
            pd.concat(dfs, names=["run_id"])
            .reset_index()
            .set_index(["run_id", "time"])
            .sort_index()
            .astype("float32[pyarrow]")
        )
        ts = remove_outliers(ts, self.timeseries_metadata)
        return ts

    @staticmethod
    def clean_timeseries_metadata() -> DataFrame:
        r"""Create DataFrame with metadata for the timeseries."""
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
                "variable"       : "string[pyarrow]",
                "lower_bound"    : "float32[pyarrow]",
                "upper_bound"    : "float32[pyarrow]",
                "lower_inclusive": "bool[pyarrow]",
                "upper_inclusive": "bool[pyarrow]",
                "unit"           : "string[pyarrow]",
                "description"    : "string[pyarrow]",
            },
            "index": "variable",
        }  # fmt: skip
        return make_dataframe(**TIMESERIES_METADATA)

    def download_file(self, fname: str, /) -> None:
        r"""Download the dataset."""
        self.LOGGER.info("Copying data files into %s.", self.rawdata_paths[fname])
        if __package__ is None:
            raise ValueError(f"Unexpected package: {__package__=}")
        with resources.path(__package__, fname) as path:
            shutil.copy(path, self.rawdata_paths[fname])
