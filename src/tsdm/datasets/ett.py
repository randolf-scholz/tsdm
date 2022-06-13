"""Electricity Transformer Dataset (ETDataset).

**Source:** https://github.com/zhouhaoyi/ETDataset
"""

__all__ = [
    # Classes
    "ETT"
]

from pathlib import Path
from typing import Literal

from pandas import read_csv

from tsdm.datasets.base import MultiFrameDataset


class ETT(MultiFrameDataset):
    r"""ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa

    BASE_URL: str = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small/"
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL: str = r"https://github.com/zhouhaoyi/ETDataset"
    r"""HTTP address containing additional information about the dataset."""
    KEYS = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""The type of the index."""
    index: list[KEYS] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    r"""IDs of the stored data-objects."""
    rawdata_files = {key: f"{key}.csv" for key in index}
    r"""Files containing the raw data."""
    rawdata_paths: dict[KEYS, Path]
    r"""Paths to the raw data."""

    def _clean(self, key: KEYS) -> None:
        r"""Create DataFrame from the .csv file."""
        df = read_csv(
            self.rawdata_paths[key], parse_dates=[0], index_col=0, dtype="float32"
        )
        return df
