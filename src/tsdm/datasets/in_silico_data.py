r"""In silico experiments.

TODO: Module Summary
"""

__all__ = [
    # Classes
    "InSilicoData",
]

import shutil
from functools import cached_property
from importlib import resources
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from tsdm.datasets import examples
from tsdm.datasets.base import SingleFrameDataset


class InSilicoData(SingleFrameDataset):
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

    rawdata_files = "in_silico.zip"

    @cached_property
    def rawdata_paths(self) -> Path:
        r"""Path to the raw data files."""
        with resources.path(examples, "in_silico.zip") as path:
            return path

    def _clean(self) -> None:
        with ZipFile(self.rawdata_paths) as files:
            dfs = {}
            for fname in files.namelist():
                key = int(fname.split(".csv")[0])
                with files.open(fname) as file:
                    df = pd.read_csv(file, index_col=0, parse_dates=[0])
                df = df.rename_axis(index="time")
                df["DOTm"] /= 100
                df.name = key
                dfs[key] = df
            df = pd.concat(dfs, names=["run_id"])
            df = df.reset_index()
            df.to_feather(self.dataset_paths)

    def _load(self) -> pd.DataFrame:
        df = pd.read_feather(self.dataset_paths)
        return df.set_index(["run_id", "time"])

    def _download(self) -> None:
        r"""Download the dataset."""
        with resources.path(examples, "examples/in_silico.zip") as path:
            shutil.copy(path, self.RAWDATA_DIR)
