r"""In silico experiments.

TODO: Module Summary
"""

__all__ = [
    # Classes
    "InSilicoData",
]


import logging
import shutil
from functools import cached_property
from importlib import resources
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from tsdm.datasets import examples
from tsdm.datasets.base import SimpleDataset

__logger__ = logging.getLogger(__name__)


class InSilicoData(SimpleDataset):
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

    @cached_property
    def rawdata_files(
        self,
    ) -> Path:
        r"""Path to the raw data files."""
        return resources.path(examples, "in_silico.zip").__enter__()

    def _clean(self) -> None:
        with ZipFile(self.rawdata_files, "r") as files:
            dfs = {}
            for fname in files.namelist():
                key = int(fname.split(".csv")[0])
                with files.open(fname, "r") as file:
                    df = pd.read_csv(file, index_col=0, parse_dates=[0])
                df = df.rename_axis(index="time")
                df["DOTm"] /= 100
                df.name = key
                dfs[key] = df
            df = pd.concat(dfs, names=["run_id"])
            df = df.reset_index()
            df.to_feather(self.dataset_files)

    def _load(self) -> pd.DataFrame:
        df = pd.read_feather(self.dataset_files)
        return df.set_index(["run_id", "time"])

    def _download(self) -> None:
        r"""Download the dataset."""
        with resources.path(examples, "in_silico.zip") as path:
            shutil.copy(path, self.rawdata_dir)
