r"""Dataset Import Facilities.

Datasets
========

Basic Usage
-----------

.. code-block:: python

   from tsdm.dataset import Electricity

   print(vars(Electricity))
   Electricity.download()
   Electricity.preprocess()
   x = Electricity.load()

   # or, simply:
   x = Electricity.dataset


Some design decisions:

1. Why not use :class:`pandas.Series` instead of Mapping for dataset?
    - ``Series[object]`` has bad performance issues in construction.

2. Should we have Dataset style iteration or dict style iteration?
    - Note that for dict, iter(dict) iterates over index.
    - For Series, DataFrame, TorchDataset, __iter__ iterates over values.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "BaseDataset",
    "BaseDatasetMetaClass",
    "SimpleDataset",
    "Dataset",
]

import logging
import os
import subprocess
import webbrowser
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Collection, Mapping, MutableMapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Optional, Union
from urllib.parse import urlparse

from pandas import DataFrame, Series

from tsdm.config import DATASETDIR, RAWDATADIR
from tsdm.util import flatten_nested, paths_exists
from tsdm.util.decorators import wrap_func
from tsdm.util.types import KeyType, NullableNestedType

__logger__ = logging.getLogger(__name__)


DATASET_OBJECT = Union[Series, DataFrame]
r"""Type hint for pandas objects"""


class BaseDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args, **kwargs):
        cls.name = cls.__name__
        if os.environ.get("GENERATING_DOCS", False):
            cls.rawdata_dir = Path(f"~/.tsdm/rawdata/{cls.name}/")
            cls.dataset_dir = Path(f"~/.tsdm/datasets/{cls.name}/")
        else:
            cls.rawdata_dir = RAWDATADIR / cls.name
            cls.dataset_dir = DATASETDIR / cls.name
        super().__init__(*args, **kwargs)


class BaseDataset(ABC, metaclass=BaseDatasetMetaClass):
    r"""Abstract base class that all dataset must subclass.

    Implements methods that are available for all dataset classes.
    """

    __slots__ = ()

    base_url: Optional[str] = None
    r"""HTTP address from where the dataset can be downloaded"""
    info_url: Optional[str] = None
    r"""HTTP address containing additional information about the dataset"""
    _initialized: bool = False
    r"""Whether the dataset has been initialized or not."""
    rawdata_dir: Path
    r"""Location where the raw data is stored."""
    dataset_dir: Path
    r"""Location where the pre-processed data is stored."""
    name: str
    r"""Name of the dataset."""

    def __new__(cls) -> BaseDataset:
        """Create a new dataset instance.

        Parameters
        ----------
        args
            Positional arguments.
        kwargs
            Keyword arguments.

        Returns
        -------
        Dataset
            A new dataset instance.
        """
        cls.rawdata_dir.mkdir(parents=True, exist_ok=True)
        cls.dataset_dir.mkdir(parents=True, exist_ok=True)
        return super().__new__(cls)

    def __init__(self, *, initialize: bool = True, reset: bool = False):
        if reset:
            self.clean()
        if initialize:
            self.load()

    def __init_subclass__(cls, *args, **kwargs):
        r"""Add wrapper code."""
        cls.load = wrap_func(
            cls.load, before=cls._load_pre_hook, after=cls._load_post_hook
        )
        cls.clean = wrap_func(
            cls.clean, before=cls._load_pre_hook, after=cls._load_post_hook
        )
        cls.download = wrap_func(
            cls.clean, before=cls._download_pre_hook, after=cls._download_post_hook
        )

    @cached_property
    def dataset(self):
        r"""Store cached version of dataset."""
        return self.load()

    # @cached_property
    # def rawdata_dir(self) -> Path:
    #     r"""Location where the raw data is stored."""
    #     if os.environ.get("GENERATING_DOCS", False):
    #         return Path(f"~/.tsdm/rawdata/{self.name}/")
    #     path = RAWDATADIR.joinpath(self.name)
    #     path.mkdir(parents=True, exist_ok=True)
    #     return path

    # @cached_property
    # def dataset_dir(self) -> Path:
    #     r"""Location where the pre-processed data is stored."""
    #     if os.environ.get("GENERATING_DOCS", False):
    #         return Path(f"~/.tsdm/datasets/{self.name}/")
    #     path = DATASETDIR.joinpath(self.name)
    #     path.mkdir(parents=True, exist_ok=True)
    #     return path

    @property
    @abstractmethod
    def dataset_files(self) -> NullableNestedType[Path]:
        r"""Location where the pre-processed data is stored."""

    @property
    @abstractmethod
    def rawdata_files(self) -> NullableNestedType[Path]:
        r"""Raw data file(s) that need to be acquired."""

    def rawdata_files_exist(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.rawdata_files)

    def dataset_files_exist(self) -> bool:
        r"""Check if dataset files exist."""
        return paths_exists(self.dataset_files)

    @abstractmethod
    def clean(self):
        r"""Clean an already downloaded raw dataset and stores it in hdf5 format.

        The cleaned dataset will be stored under the path specified in `cls.dataset_files`.

        .. code-block:: python

            cls.dataset_files = DATASETDIR  / f"{cls.__name__}.h5"

        .. note::
            Must be implemented for any dataset class!!
        """

    @abstractmethod
    def load(self):
        r"""Load the pre-processed dataset.

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

    def _load_pre_hook(self, **_: Any) -> None:
        r"""Code that is executed before ``load``."""
        __logger__.debug("%s: START LOADING.", self.name)

    def _load_post_hook(self, **_: Any) -> None:
        r"""Code that is executed after ``load``."""
        __logger__.debug("%s: DONE LOADING.", self.name)

    def _clean_pre_hook(self, **_: Any) -> None:
        r"""Code that is executed before ``clean``."""
        __logger__.debug("%s: START CLEANING.", self.name)

    def _clean_post_hook(self, **_: Any) -> None:
        r"""Code that is executed after ``clean``."""
        __logger__.debug("%s: DONE CLEANING.", self.name)

    def _download_pre_hook(self, **_: Any) -> None:
        r"""Code that is executed before ``download``."""
        __logger__.debug("%s: START DOWNLOADING.", self.name)

    def _download_post_hook(self, **_: Any) -> None:
        r"""Code that is executed after ``download``."""
        __logger__.debug("%s: DONE DOWNLOADING.", self.name)

    def download(self, *, url: Optional[Union[str, Path]] = None) -> None:
        r"""Download the dataset and stores it in `cls.rawdata_dir`.

        The default downloader checks if

        1. The url points to kaggle.com => uses `kaggle competition download`
        2. The url points to github.com => checkout directory with `svn`
        3. Else simply use `wget` to download the `cls.url` content,

        Overwrite if you need custom downloader
        """
        if url is None and self.base_url is None:
            __logger__.info("%s: Dataset provides no url. Assumed offline", self.name)
            return

        target_url: str = str(self.base_url) if url is None else str(url)
        parsed_url = urlparse(target_url)

        __logger__.info("%s: Obtaining dataset from %s", self.name, target_url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                f"kaggle competitions download -p {self.rawdata_dir} -c {kaggle_name}",
                shell=True,
                check=True,
            )
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                f"svn export --force {target_url.replace('tree/main', 'trunk')} {self.rawdata_dir}",
                shell=True,
                check=True,
            )
        else:  # default parsing, including for UCI dataset
            cut_dirs = target_url.count("/") - 3
            subprocess.run(
                f"wget -r -np -nH -N --cut-dirs {cut_dirs} -P '{self.rawdata_dir}' {target_url}",
                shell=True,
                check=True,
            )

        __logger__.info("%s: Finished importing dataset from %s", self.name, target_url)

    def info(self):
        r"""Open dataset information in browser."""
        if self.info_url is None:
            print(self.__doc__)
        else:
            webbrowser.open_new_tab(self.info_url)

    def __len__(self):
        r"""Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        r"""Return the sample at index `idx`."""
        return self.dataset[idx]

    def __setitem__(self, key, value):
        r"""Return the sample at index `idx`."""
        self.dataset[key] = value

    def __iter__(self):
        r"""Return an iterator over the dataset."""
        return iter(self.dataset)

    def __repr__(self):
        r"""Return a string representation of the dataset."""
        return f"{self.name}\n{self.dataset}"

    def _repr_html_(self):
        if hasattr(self.dataset, "_repr_html_"):
            header = f"<h3>{self.name}</h3>"
            html_repr = self.dataset._repr_html_()  # pylint: disable=protected-access
            return header + html_repr
        raise NotImplementedError


##########################################
# Dataset consisting of Single DataFrame #
##########################################


class SimpleDataset(BaseDataset):
    r"""Dataset class that consists of a singular DataFrame."""

    @cached_property
    def dataset_files(self) -> NullableNestedType[Path]:
        r"""Return the path to the dataset files."""
        return self.dataset_dir / f"{self.name}.feather"

    @cached_property
    def dataset(self) -> DATASET_OBJECT:
        r"""Store cached version of dataset."""
        return self.load()

    @abstractmethod
    def _clean(self) -> None:
        r"""Clean the dataset."""

    @abstractmethod
    def _load(self) -> DATASET_OBJECT:
        r"""Load the dataset."""

    def clean(self, *, force: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        if self.dataset_files_exist() and not force:
            __logger__.debug("%s: Dataset files already exist, skipping.", self.name)
            return
        if not self.rawdata_files_exist():
            self.download()
        self._clean()

    def load(self) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        if not self.rawdata_files_exist():
            self.download()
        else:
            __logger__.debug("%s: Rawdata files already exist!", self.name)

        if not self.dataset_files_exist():
            self.clean()
        else:
            __logger__.debug("%s: Dataset files already exist!", self.name)

        return self._load()

    def __getattr__(self, key):
        """Attribute lookup."""
        if key in self.dataset:
            return self.dataset[key]
        if hasattr(self.dataset, key):
            return getattr(self.dataset, key)
        raise AttributeError(f"{self.name} has no attribute {key}")

    #
    # def __repr__(self):
    #     r"""Return a string representation of the dataset."""
    #     return f"{self.name}\n{self.dataset}"
    #
    # def _repr_html_(self):
    #     # if hasattr(self.dataset, "_repr_html_"):
    #     header = f"<h3>{self.name}</h3>"
    #     return header + self.dataset._repr_html_()
    #     # raise NotImplementedError


#############################################
# Dataset consisting of Multiple DataFrames #
#############################################


class Dataset(BaseDataset, Mapping, Generic[KeyType]):
    r"""Dataset class that consists of a multiple DataFrames.

    The Datasets are accessed by their index.
    We subclass Mapping to provide the mapping interface.
    """

    __slots__ = ()  # Do we need this?

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyType]:
        r"""Return the index of the dataset."""
        # implement loading of dataset

    @cached_property
    def dataset(self) -> MutableMapping[KeyType, DATASET_OBJECT]:
        r"""Store cached version of dataset."""
        return {key: None for key in self.index}

    @cached_property
    def dataset_files(self) -> Mapping[KeyType, Path]:
        r"""Return the paths of the dataset files."""
        return {key: self.dataset_dir / f"{key}.feather" for key in self.index}

    def __getattr__(self, key):
        """Attribute lookup for index."""
        if self.index is not None and key in self.index:
            if self.dataset[key] is None:
                self.load(key=key)
            return self.dataset[key]
        raise AttributeError(f"No attribute {key} in {self.name}")

    def __repr__(self):
        r"""Pretty Print."""
        if len(self.index) > 6:
            indices = list(self.index)
            selection = [str(indices[k]) for k in [0, 1, 2, -2, -1]]
            selection[2] = "..."
            return f"{self.name}({', '.join(selection)})"
        return f"{self.name}{self.index}"

    def rawdata_files_exist(self, key: Optional[KeyType] = None) -> bool:
        r"""Check if raw data files exist."""
        if not isinstance(self.rawdata_files, Mapping) or key is None:
            return paths_exists(self.rawdata_files)

        assert isinstance(self.rawdata_files, Mapping)
        return paths_exists(self.rawdata_files[key])

    def dataset_files_exist(self, key: Optional[KeyType] = None) -> bool:
        r"""Check if dataset files exist."""
        if key is None:
            return paths_exists(self.dataset_files)

        assert isinstance(self.dataset_files, Mapping)
        return paths_exists(self.dataset_files[key])

    def _clean_pre_hook(self, *, key: Optional[KeyType] = None, **_: Any) -> None:
        r"""Code that is executed before ``clean``."""
        __logger__.debug("%s/%s: START cleaning dataset!", self.name, key)

    def _clean_post_hook(self, *, key: Optional[KeyType] = None, **_: Any) -> None:
        r"""Code that is executed after ``clean``."""
        __logger__.debug("%s/%s: DONE cleaning dataset!", self.name, key)

    @abstractmethod
    def _clean(self, key: KeyType) -> None:
        r"""Clean the selected DATASET_OBJECT."""

    def clean(self, key: Optional[KeyType] = None, force: bool = False) -> None:
        r"""Clean the selected DATASET_OBJECT.

        Parameters
        ----------
        key: Optional[KeyType] = None
            The key of the dataset to clean. If None, clean all dataset.
        force: bool = False
            Force cleaning of dataset.
        """
        if not self.rawdata_files_exist(key=key):
            __logger__.debug("%s/%s missing, fetching it now!", self.name, key)
            self.download(key=key, force=force)

        if (
            key in self.dataset_files
            and self.dataset_files_exist(key=key)
            and not force
        ):
            __logger__.debug("%s/%s already exists, skipping.", self.name, key)
            return

        if key is None:
            for key_ in self.index:
                self.clean(key=key_, force=force)
            return

        self._clean(key=key)

    def _load_pre_hook(self, *, key: Optional[KeyType] = None, **_: Any) -> None:
        r"""Code that is executed before ``load``."""
        __logger__.debug("%s/%s: START loading dataset!", self.name, key)

    def _load_post_hook(self, *, key: Optional[KeyType] = None, **_: Any) -> None:
        r"""Code that is executed after ``load``."""
        __logger__.debug("%s/%s: DONE loading dataset!", self.name, key)

    @abstractmethod
    def _load(self, key: KeyType) -> DATASET_OBJECT:
        r"""Clean the selected DATASET_OBJECT."""

    def load(
        self, *, key: Optional[KeyType] = None, force: bool = False, **kwargs: Any
    ) -> Union[DATASET_OBJECT, Mapping[KeyType, DATASET_OBJECT]]:
        r"""Load the selected DATASET_OBJECT.

        Parameters
        ----------
        key: Optional[KeyType] = None
        force: bool = False
            Reload the dataset if it already exists.

        Returns
        -------
        DATASET_OBJECT | Mapping[KeyType, DATASET_OBJECT]
        """
        if not self.rawdata_files_exist(key=key):
            self.download(key=key, force=force, **kwargs)

        if not self.dataset_files_exist(key=key):
            self.clean(key=key, force=force)

        if key is None:
            return {k: self.load(key=k, force=force, **kwargs) for k in self.index}

        if key in self.dataset and self.dataset[key] is not None and not force:
            __logger__.debug("%s/%s: dataset already exists, skipping!", self.name, key)
            return self.dataset[key]

        self.dataset[key] = self._load(key=key)
        return self.dataset[key]

    def _download(self, *, key: KeyType) -> None:
        r"""Download the selected DATASET_OBJECT."""
        assert key is not None, "Called _download with key=None!"

        if self.base_url is None:
            __logger__.debug("%s: Dataset provides no url. Assumed offline", self.name)
            return

        if self.rawdata_files is None:
            super().download()
            return

        if isinstance(self.rawdata_files, Mapping):
            files: Union[None, Path, Collection[Path]] = self.rawdata_files[key]
        else:
            files = self.rawdata_files

        for path in flatten_nested(files, kind=Path):
            super().download(url=self.base_url + path.name)

    def _download_pre_hook(self, *, key: Optional[KeyType] = None, **_: Any) -> None:
        r"""Code that is executed before ``download``."""
        __logger__.debug("%s/%s: START downloading dataset!", self.name, key)

    def _download_post_hook(self, *, key: Optional[KeyType] = None, **_: Any) -> None:
        r"""Code that is executed after ``download``."""
        __logger__.debug("%s/%s: DONE downloading dataset!", self.name, key)

    def download(
        self,
        *,
        key: Optional[KeyType] = None,
        url: Optional[Union[str, Path]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Download the dataset.

        Parameters
        ----------
        key: Optional[KeyType] = None
        url: Optional[Union[str, Path]] = None
        force: bool = False
            Force re-downloading of dataset.
        """
        if self.base_url is None:
            __logger__.debug(
                "%s: Dataset provides no base_url. Assumed offline", self.name
            )
            return

        if not force and self.rawdata_files_exist(key=key):
            __logger__.debug(
                "%s/%s: Rawdata files already exist, skipping.",
                self.name,
                "" if key is None else key,
            )
            return

        if key is None:
            if isinstance(self.rawdata_files, Mapping):
                for key_ in self.rawdata_files:
                    self.download(key=key_, url=url, force=force, **kwargs)
            else:
                super().download(url=url)
            return

        self._download(key=key)
