"""Electricity Transformer Dataset (ETDataset).

**Source:** https://github.com/zhouhaoyi/ETDataset
"""

import logging
from pathlib import Path
from typing import Final

from pandas import DataFrame, read_csv, read_hdf

from tsdm.datasets.dataset import BaseDataset

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


class ETTh1(BaseDataset):
    """ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa

    url: str = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small"
    info_url: str = r"https://github.com/zhouhaoyi/ETDataset"
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def clean(cls):
        r"""Create DataFrame from the .csv file."""
        LOGGER.info("Cleaning dataset '%s'", cls.__name__)

        filename = "ETTh1.csv"
        with open(cls.rawdata_path.joinpath(filename), "r") as file:
            df = read_csv(file, parse_dates=[0], index_col=0)
        df.name = cls.__name__

        # Store the preprocessed dataset as h5 file
        df.to_hdf(cls.dataset_file, key=cls.__name__)

        LOGGER.info("Finished cleaning dataset '%s'", cls.__name__)

    @classmethod
    def load(cls):
        r"""Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        df = read_hdf(cls.dataset_file, key=cls.__name__)
        df = DataFrame(df)
        return df


class ETTh2(BaseDataset):
    """ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa

    url: str = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small"
    info_url: str = r"https://github.com/zhouhaoyi/ETDataset"
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def clean(cls):
        r"""Create DataFrame from the .csv file."""
        LOGGER.info("Cleaning dataset '%s'", cls.__name__)

        filename = "ETTm2.csv"
        with open(cls.rawdata_path.joinpath(filename), "r") as file:
            df = read_csv(file, parse_dates=[0], index_col=0)
        df.name = cls.__name__

        # Store the preprocessed dataset as h5 file
        df.to_hdf(cls.dataset_file, key=cls.__name__)

        LOGGER.info("Finished cleaning dataset '%s'", cls.__name__)

    @classmethod
    def load(cls):
        r"""Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        df = read_hdf(cls.dataset_file, key=cls.__name__)
        df = DataFrame(df)
        return df


class ETTm1(BaseDataset):
    """ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa

    url: str = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small"
    info_url: str = r"https://github.com/zhouhaoyi/ETDataset"
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def clean(cls):
        r"""Create DataFrame from the .csv file."""
        LOGGER.info("Cleaning dataset '%s'", cls.__name__)

        filename = "ETTm1.csv"
        with open(cls.rawdata_path.joinpath(filename), "r") as file:
            df = read_csv(file, parse_dates=[0], index_col=0)
        df.name = cls.__name__

        # Store the preprocessed dataset as h5 file
        df.to_hdf(cls.dataset_file, key=cls.__name__)

        LOGGER.info("Finished cleaning dataset '%s'", cls.__name__)

    @classmethod
    def load(cls):
        r"""Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        df = read_hdf(cls.dataset_file, key=cls.__name__)
        df = DataFrame(df)
        return df


class ETTm2(BaseDataset):
    """ETT-small-h1.

    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    | Field       | date              | HUFL             | HULL              | MUFL               | MULL                | LUFL            | LULL             | OT                       |
    +=============+===================+==================+===================+====================+=====================+=================+==================+==========================+
    | Description | The recorded date | High UseFul Load | High UseLess Load | Middle UseFul Load | Middle UseLess Load | Low UseFul Load | Low UseLess Load | Oil Temperature (target) |
    +-------------+-------------------+------------------+-------------------+--------------------+---------------------+-----------------+------------------+--------------------------+
    """  # pylint: disable=line-too-long # noqa

    url: str = r"https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small"
    info_url: str = r"https://github.com/zhouhaoyi/ETDataset"
    dataset: DataFrame
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def clean(cls):
        r"""Create DataFrame from the .csv file."""
        LOGGER.info("Cleaning dataset '%s'", cls.__name__)

        filename = "ETTm2.csv"
        with open(cls.rawdata_path.joinpath(filename), "r") as file:
            df = read_csv(file, parse_dates=[0], index_col=0)
        df.name = cls.__name__

        # Store the preprocessed dataset as h5 file
        df.to_hdf(cls.dataset_file, key=cls.__name__)

        LOGGER.info("Finished cleaning dataset '%s'", cls.__name__)

    @classmethod
    def load(cls):
        r"""Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        df = read_hdf(cls.dataset_file, key=cls.__name__)
        df = DataFrame(df)
        return df
