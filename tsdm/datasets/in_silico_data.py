r"""
In silico experiments
"""  # pylint: disable=line-too-long # noqa

import logging
from importlib import resources
from pathlib import Path

import h5py
import pandas as pd
from pandas import DataFrame

from .dataset import BaseDataset, DatasetMetaClass
from .examples import in_silico

logger = logging.getLogger(__name__)


class InSilicoData(BaseDataset, metaclass=DatasetMetaClass):
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
    """  # pylint: disable=line-too-long # noqa

    url:          str = ""
    dataset:      dict[str, DataFrame]
    rawdata_path: Path
    dataset_path: Path
    dataset_file: Path

    @classmethod
    def clean(cls):
        """Creates DataFrame with 1 column per client and :class:`pandas.DatetimeIndex`
        """
        dataset = cls.__name__
        logger.info("Cleaning dataset '%s'", dataset)

        dfs = {}
        for resource in resources.contents(in_silico):
            if resource.split('.')[-1] != "txt":
                continue
            with resources.path(in_silico, resource) as path:
                with open(path, 'r') as file:
                    df = pd.read_csv(file, index_col=0)
                    df = df.rename_axis(index="time")
                    df["DOTm"] /= 100
                    df.name = "".join([c for c in file.name if c.isdigit()])
                    dfs[df.name] = df

        if cls.dataset_file.exists():
            cls.dataset_file.unlink()

        for df in dfs.values():
            df.to_hdf(cls.dataset_file, key=df.name, mode='a')

        logger.info("Finished cleaning dataset '%s'", dataset)

    @classmethod
    def load(cls):
        """Load the dataset from hdf-5 file"""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        with h5py.File(cls.dataset_file, "r") as file:
            read_dfs = {}
            for key in file.keys():
                read_dfs[key] = pd.read_hdf(cls.dataset_file, key=key)
            return read_dfs

    @classmethod
    def download(cls):
        """nothing to do here"""