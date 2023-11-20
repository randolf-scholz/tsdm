r"""In silico experiments."""

__all__ = ["InSilico", "InSilicoTSC"]

import shutil
from importlib import resources
from zipfile import ZipFile

import pandas as pd
from pandas import DataFrame
from typing_extensions import Literal, TypeAlias

from tsdm.data import InlineTable, make_dataframe, remove_outliers
from tsdm.datasets.base import MultiTableDataset, TimeSeriesCollection

TIMESERIES_DESCRIPTION: InlineTable = {
    "data": [
        # fmt: off
        ("Biomass"  , 0, None, True, True, "g/L", None),
        ("Substrate", 0, None, True, True, "g/L", None),
        ("Acetate"  , 0, None, True, True, "g/L", None),
        ("DOTm"     , 0, 100,  True, True, "%",   None),
        ("Product"  , 0, None, True, True, "g/L", None),
        ("Volume"   , 0, None, True, True, "L",   None),
        ("Feed"     , 0, None, True, True, "μL",  None),
        # fmt: on
    ],
    "schema": {
        # fmt: off
        "variable"       : "string[pyarrow]",
        "lower_bound"    : "float32[pyarrow]",
        "upper_bound"    : "float32[pyarrow]",
        "lower_inclusive": "bool[pyarrow]",
        "upper_inclusive": "bool[pyarrow]",
        "unit"           : "string[pyarrow]",
        "description"    : "string[pyarrow]",
        # fmt: on
    },
    "index": "variable",
}

KEY: TypeAlias = Literal[
    "timeseries",
    "timeseries_description",
]


class InSilico(MultiTableDataset[KEY, DataFrame]):
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
        "in_silico.zip": (
            "sha256:ee9ad6278fb27dd933c22aecfc7b5b2501336e859a7f012cace2bb265f713cba"
        ),
    }
    table_names = ["timeseries", "timeseries_description"]
    table_shapes = {"timeseries": (5206, 7)}

    def _timeseries(self) -> DataFrame:
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
        ts = remove_outliers(ts, self.timeseries_description)
        return ts

    def clean_table(self, key: KEY) -> DataFrame:
        if key == "timeseries":
            return self._timeseries()
        if key == "timeseries_description":
            return make_dataframe(**TIMESERIES_DESCRIPTION)
        raise KeyError(f"Unknown table {key}.")

    def download_file(self, fname: str, /) -> None:
        r"""Download the dataset."""
        self.LOGGER.info("Copying data files into %s.", self.rawdata_paths[fname])
        with resources.path(__package__, fname) as path:
            shutil.copy(path, self.rawdata_paths[fname])


class InSilicoTSC(TimeSeriesCollection):
    r"""The in silico dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = InSilico()
        super().__init__(
            timeseries=ds.timeseries,
            timeseries_description=ds.timeseries_description,
        )
