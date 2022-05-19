"""Electricity Transformer Dataset (ETDataset).

**Source:** https://github.com/zhouhaoyi/ETDataset
"""

__all__ = [
    # Classes
    "ETT"
]

from typing import Literal

from pandas import DataFrame, read_csv, read_feather

from tsdm.datasets.base import Dataset


class ETT(Dataset):
    r"""ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa

    base_url: str = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small/"
    r"""HTTP address from where the dataset can be downloaded."""
    info_url: str = r"https://github.com/zhouhaoyi/ETDataset"
    r"""HTTP address containing additional information about the dataset."""
    dataset: DataFrame
    r"""Store cached version of dataset."""
    KEYS = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""The type of the index."""
    index: list[KEYS] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""IDs of the stored data-objects."""
    rawdata_files = {key: f"{key}.csv" for key in index}
    r"""Files containing the raw data."""
    dataset_files = {key: f"{key}.feather" for key in index}

    def _clean(self, key: KEYS) -> None:
        r"""Create DataFrame from the .csv file."""
        with open(self.rawdata_files[key], "r", encoding="utf8") as file:
            df = read_csv(file, parse_dates=[0], index_col=0)
            df.name = self.__class__.__name__
            # Store the preprocessed dataset as h5 file
            df = df.reset_index()
            df.to_feather(self.dataset_files[key])

    def _load(self, key: KEYS) -> DataFrame:
        r"""Load the dataset from hdf-5 file."""
        df = read_feather(self.dataset_files[key])
        return df.set_index("date")
