"""Electricity Transformer Dataset (ETDataset).

**Source:** https://github.com/zhouhaoyi/ETDataset
"""

__all__ = [
    # Classes
    "ETT"
]

import logging
from functools import cached_property
from pathlib import Path
from typing import Literal

from pandas import DataFrame, read_csv, read_feather

from tsdm.datasets.base import Dataset

__logger__ = logging.getLogger(__name__)


class ETT(Dataset):
    r"""ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa

    base_url: str = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small"
    r"""HTTP address from where the dataset can be downloaded."""
    info_url: str = r"https://github.com/zhouhaoyi/ETDataset"
    r"""HTTP address containing additional information about the dataset."""
    dataset: DataFrame
    r"""Store cached version of dataset."""
    KEYS = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""The type of the index."""
    index: list[KEYS] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""IDs of the stored data-objects."""

    @cached_property
    def rawdata_files(self) -> dict[KEYS, Path]:
        r"""Path of the raw data file."""
        return {key: self.rawdata_dir / f"{key}.csv" for key in self.index}

    def _clean(self, key: KEYS) -> None:
        r"""Create DataFrame from the .csv file."""
        with open(self.rawdata_files[key], "r", encoding="utf8") as file:
            df = read_csv(file, parse_dates=[0], index_col=0)
            df.name = self.name
            # Store the preprocessed dataset as h5 file
            df = df.reset_index()
            df.to_feather(self.dataset_files[key])

    def _load(self, key: KEYS) -> DataFrame:
        r"""Load the dataset from hdf-5 file."""
        df = read_feather(self.dataset_files[key])
        return df.set_index("date")
