r"""
Physionet challenge 2019
"""


import logging
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import numpy as np
from pandas import DataFrame, read_csv, read_hdf

from tsdm.datasets.dataset import BaseDataset

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["Physionet2019"]



class Physionet2019(BaseDataset):
    r"""
    Physionet challenge 2019
    """

    url: str = r"https://archive.physionet.org/users/shared/challenge-2019/"
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

