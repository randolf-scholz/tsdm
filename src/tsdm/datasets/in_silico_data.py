r"""In silico experiments."""

__all__ = [
    # Classes
    "InSilicoData",
]

import shutil
from functools import cached_property
from importlib import resources
from zipfile import ZipFile

import pandas as pd

from tsdm.datasets import examples
from tsdm.datasets.base import SingleTableDataset


class InSilicoData(SingleTableDataset):
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

    DATASET_HASH = "f6938b4e9de35824c24c3bdc7f08c4d9bfcf9272eaeb76f579d823ca8628bff0"
    DATASET_SHAPE = (5206, 7)
    TABLE_HASH = 652930435272677160

    rawdata_files = ["in_silico.zip"]
    rawdata_hashes = {
        "in_silico.zip": "sha256:ee9ad6278fb27dd933c22aecfc7b5b2501336e859a7f012cace2bb265f713cba",
    }

    @cached_property
    def index(self) -> pd.Index:
        r"""Return the index of the dataset."""
        return self.timeseries.index.get_level_values(0).unique()

    @cached_property
    def timeseries(self) -> pd.DataFrame:
        r"""Return the timeseries of the dataset."""
        return self.table

    def clean_table(self) -> None:
        with ZipFile(str(self.rawdata_paths)) as files:
            dfs = {}
            for fname in files.namelist():
                key = int(fname.split(".csv")[0])
                with files.open(fname) as file:
                    df = pd.read_csv(file, index_col=0, parse_dates=[0])
                df = df.rename_axis(index="time")
                df["DOTm"] /= 100
                dfs[key] = df
        ds = pd.concat(dfs, names=["run_id"])
        ds = ds.reset_index()
        ds = ds.set_index(["run_id", "time"])
        ds = ds.sort_values(by=["run_id", "time"])
        ds = ds.astype("Float32")
        return ds

    def download_all_files(self) -> None:
        r"""Download the dataset."""
        self.LOGGER.info("Copying data files into %s.", self.rawdata_paths)
        with resources.path(examples, self.rawdata_files) as path:
            shutil.copy(path, str(self.rawdata_paths))
