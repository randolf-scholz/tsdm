r"""In silico experiments.

TODO: Module Summary
"""

__all__ = [
    # Classes
    "InSilicoData",
]

import logging
from functools import cache
from importlib import resources
from pathlib import Path

import h5py
import pandas as pd

from tsdm.datasets.base import BaseDataset
from tsdm.datasets.examples import in_silico

__logger__ = logging.getLogger(__name__)


class InSilicoData(BaseDataset):
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

    @classmethod
    @property
    @cache
    def dataset_file(cls) -> Path:
        r"""Path of the dataset file."""
        return cls.dataset_path.joinpath(f"{cls.__name__}.h5")

    @classmethod
    def clean(cls):
        """Create `DataFrame` with 1 column per client and `DatetimeIndex`."""
        dataset = cls.__name__
        __logger__.info("Cleaning dataset '%s'", dataset)

        dfs = {}
        for resource in resources.contents(in_silico):
            if resource.split(".")[-1] != "txt":
                continue
            with resources.path(in_silico, resource) as path:
                with open(path, "r", encoding="utf8") as file:
                    df = pd.read_csv(file, index_col=0, parse_dates=[0])
                    df = df.rename_axis(index="time")
                    df["DOTm"] /= 100
                    df.name = "run_" + "".join([c for c in file.name if c.isdigit()])
                    dfs[df.name] = df

        if cls.dataset_file.exists():
            cls.dataset_file.unlink()

        for df in dfs.values():
            df.to_hdf(cls.dataset_file, key=df.name, mode="a")

        __logger__.info("Finished cleaning dataset '%s'", dataset)

    @classmethod
    def load(cls):
        r"""Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        with h5py.File(cls.dataset_file, "r") as file:
            read_dfs = {}
            for key in file.keys():
                read_dfs[key] = pd.read_hdf(cls.dataset_file, key=key)
            return read_dfs
