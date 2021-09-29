r"""Dataset Import Facilities.

Datasets
========

Basic Usage
-----------

.. code-block:: python

   from tsdm.datasets import Electricity

   print(vars(Electricity))
   Electricity.download()
   Electricity.preprocess()
   x = Electricity.load()

   # or, simply:
   x = Electricity.dataset
"""

from __future__ import annotations

__all__ = [
    # Classes
    "BaseDataset",
    "DatasetMetaClass",
    "SequenceDataset",
]


import logging
import os
import subprocess
from abc import ABC, ABCMeta, abstractmethod
from functools import cache
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import torch
from torch import Tensor

from tsdm.config import DATASETDIR, RAWDATADIR

LOGGER = logging.getLogger(__name__)


class DatasetMetaClass(ABCMeta):
    r"""Dataset metaclass providing class attributes.

    This metaclass makes sure that any dataset class has certain attributes like
    `rawdata_path`, even before being initialized. As a consequence, the dataset classes
    generally do not need to be initialized.

    Attributes
    ----------
    url: str
        a http address from where the dataset can be downloaded
    dataset: Series, DataFrame, DataArray or Dataset
        internal storage of the dataset
    rawdata_path: Path
        location where the raw data is stored
    dataset_path: Path
        location where the pre-processed data is stored
    dataset_file: Path
    """

    @property  # type: ignore
    @cache
    def dataset(cls):
        r"""Store cached version of dataset."""
        # What is the best practice for metaclass methods that call each other?
        # https://stackoverflow.com/q/47615318/9318372
        if os.environ.get("GENERATING_DOCS", False):
            return "the dataset"
        return cls.load()  # pylint: disable=E1120

    @property
    def rawdata_path(cls):
        r"""Location where the raw data is stored."""
        if os.environ.get("GENERATING_DOCS", False):
            return Path(f"~/.tsdm/rawdata/{cls.__name__}/")
        RAWDATADIR.mkdir(parents=True, exist_ok=True)
        return RAWDATADIR.joinpath(cls.__name__)

    @property
    def dataset_path(cls):
        r"""Location where the pre-processed data is stored."""
        if os.environ.get("GENERATING_DOCS", False):
            return Path("~/.tsdm/datasets/")
        DATASETDIR.mkdir(parents=True, exist_ok=True)
        return DATASETDIR

    @property
    def dataset_file(cls):
        r"""Path of the dataset file."""
        if os.environ.get("GENERATING_DOCS", False):
            return Path(f"~/.tsdm/datasets/{cls.__name__}.h5")
        return DATASETDIR.joinpath(f"{cls.__name__}.h5")

    @abstractmethod
    def load(cls):
        r"""Load the dataset."""

    @abstractmethod
    def download(cls):
        r"""Download the dataset."""

    @abstractmethod
    def clean(cls):
        r"""Clean the dataset."""


class BaseDataset(ABC, metaclass=DatasetMetaClass):
    r"""Abstract base class that all datasets must subclass.

    Implements methods that are available for all dataset classes.
    """

    url: Union[str, None] = None
    """a http address from where the dataset can be downloaded"""
    dataset = classmethod(DatasetMetaClass.dataset)  # type: ignore
    """The dataset cached"""
    rawdata_path: Path = classmethod(DatasetMetaClass.rawdata_path)  # type: ignore
    """location where the raw data is stored"""
    dataset_path: Path = classmethod(DatasetMetaClass.dataset_path)  # type: ignore
    """location where the pre-processed data is stored"""
    dataset_file: Path = classmethod(DatasetMetaClass.dataset_file)  # type: ignore
    """The dataset file"""

    @classmethod
    @abstractmethod
    def clean(cls):
        r"""Clean an already downloaded raw dataset and stores it in hdf5 format.

        The cleaned dataset will be stored under the path specified in `cls.dataset_file`.

        .. code-block:: python

            cls.dataset_file = DATASETDIR.joinpath(F"{cls.__name__}.h5")

        .. note::
            Must be implemented for any dataset class!!
        """

    @classmethod
    @abstractmethod
    def load(cls):
        r"""Load the dataset stored in hdf5 format in the path `cls.dataset_file`.

        Use the following template for dataset classes:

        .. code-block:: python

                @classmethod
                def load(cls):
                    super().load()  # <- makes sure DS is downloaded and preprocessed
                    ...
                    return dataset

        .. note::
            Must be implemented for any dataset class!!
        """
        if not cls.dataset_file.exists():
            cls.download()
            cls.clean()

    @classmethod
    def download(cls):
        r"""Download the dataset and stores it in `cls.rawdata_path`.

        The default downloader checks if

        1. The url points to kaggle.com => uses `kaggle competition download`
        2. The url points to github.com => checkout directory with `svn`
        3. Else simply use `wget` to download the `cls.url` content,

        Overwrite if you need custom downloader
        """
        if cls.url is None:
            LOGGER.info("Dataset '%s' provides no url. Assumed offline", cls.__name__)
            return

        dataset = cls.__name__
        parsed_url = urlparse(cls.url)
        LOGGER.info("Obtaining dataset '%s' from %s", dataset, cls.url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                f"kaggle competitions download -p {cls.rawdata_path} -c {kaggle_name}",
                shell=True,
                check=True,
            )
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                f"svn export {cls.url.replace('tree/main', 'trunk')} {cls.rawdata_path}",
                shell=True,
                check=True,
            )
        else:  # default parsing, including for UCI datasets
            cut_dirs = cls.url.count("/") - 3
            subprocess.run(
                f"wget -r -np -nH -N --cut-dirs {cut_dirs} -P '{cls.rawdata_path}' {cls.url}",
                shell=True,
                check=True,
            )

        LOGGER.info("Finished importing dataset '%s' from %s", dataset, cls.url)

    @classmethod
    def to_trainloader(cls):
        """Return trainloader object."""


class SequenceDataset(torch.utils.data.Dataset):
    r"""Sequential Dataset."""

    def __init__(self, tensors: list[Tensor]):
        assert all(len(x) == len(tensors[0]) for x in tensors)
        self.tensors = tensors

    def __len__(self):
        r"""Length of the dataset."""
        return len(self.tensors[0])

    def __getitem__(self, idx):
        r"""Get the same slice from each tensor."""
        return [x[idx] for x in self.tensors]
