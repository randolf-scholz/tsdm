r"""Base Classes for dataset."""

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
from collections.abc import Mapping, MutableMapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Optional, Union, overload
from urllib.parse import urlparse

from pandas import DataFrame, Series

from tsdm.config import DATASETDIR, RAWDATADIR
from tsdm.util import flatten_nested, paths_exists, prepend_path
from tsdm.util.types import KeyType, Nested, PathType

DATASET_OBJECT = Union[Series, DataFrame]
r"""Type hint for pandas objects."""


class BaseDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args, **kwargs):
        if os.environ.get("GENERATING_DOCS", False):
            cls.rawdata_dir = Path(f"~/.tsdm/rawdata/{cls.__name__}/")
            cls.dataset_dir = Path(f"~/.tsdm/datasets/{cls.__name__}/")
        else:
            cls.rawdata_dir = RAWDATADIR / cls.__name__
            cls.dataset_dir = DATASETDIR / cls.__name__

            cls.__logger__ = logging.getLogger(f"{__package__}.{cls.__name__}")

        super().__init__(*args, **kwargs)


class BaseDataset(ABC, metaclass=BaseDatasetMetaClass):
    r"""Abstract base class that all dataset must subclass.

    Implements methods that are available for all dataset classes.
    """

    __slots__ = ()

    base_url: Optional[str] = None
    r"""HTTP address from where the dataset can be downloaded."""
    info_url: Optional[str] = None
    r"""HTTP address containing additional information about the dataset."""
    _initialized: bool = False
    r"""Whether the dataset has been initialized or not."""
    rawdata_dir: Path
    r"""Location where the raw data is stored."""
    dataset_dir: Path
    r"""Location where the pre-processed data is stored."""
    __logger__: logging.Logger
    r"""Logger for the dataset."""

    def __init__(self, *, initialize: bool = True, reset: bool = False):
        """Initialize the dataset."""
        # Create folders
        self.rawdata_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        if reset:
            self.clean()
        if initialize:
            self.load()

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
        return f"{self.__class__.__name__}\n{self.dataset}"

    @cached_property
    def dataset(self):
        r"""Store cached version of dataset."""
        return self.load()

    @property
    @abstractmethod
    def dataset_files(self) -> Nested[Optional[PathType]]:
        r"""Relative paths to the cleaned dataset file(s)."""

    @property
    @abstractmethod
    def rawdata_files(self) -> Nested[Optional[PathType]]:
        r"""Relative paths to the raw dataset file(s)."""

    @cached_property
    def rawdata_paths(self) -> Nested[Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return prepend_path(self.rawdata_files, parent=self.rawdata_dir)

    @cached_property
    def dataset_paths(self) -> Nested[Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return prepend_path(self.dataset_files, parent=self.rawdata_dir)

    def rawdata_files_exist(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.rawdata_paths)

    def dataset_files_exist(self) -> bool:
        r"""Check if dataset files exist."""
        return paths_exists(self.dataset_paths)

    @abstractmethod
    def clean(self):
        r"""Clean an already downloaded raw dataset and stores it in self.data_dir.

        Preferably, use the '.feather' data format.
        """

    @abstractmethod
    def load(self):
        r"""Load the pre-processed dataset."""

    @abstractmethod
    def download(self) -> None:
        r"""Download the raw data."""

    def download_from_url(self, url: str) -> None:
        r"""Download files from a URL."""
        self.__logger__.info("Obtaining files from %s", url)
        parsed_url = urlparse(url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                f"kaggle competitions download -p {self.rawdata_dir} -c {kaggle_name}",
                shell=True,
                check=True,
            )
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                f"svn export --force {url.replace('tree/main', 'trunk')} {self.rawdata_dir}",
                shell=True,
                check=True,
            )
        else:  # default parsing, including for UCI dataset
            cut_dirs = url.count("/") - 3
            subprocess.run(
                f"wget -r -np -nH -N --cut-dirs {cut_dirs} -P '{self.rawdata_dir}' {url}",
                shell=True,
                check=True,
            )

        self.__logger__.info("Finished importing files from %s", url)

    # def download(
    #     self, *, url: Optional[Union[str, Path]] = None, force: bool = False
    # ) -> None:
    #     r"""Download the dataset and stores it in `self.rawdata_dir`.
    #
    #     The default downloader checks if
    #
    #     1. The url points to kaggle.com => uses `kaggle competition download`
    #     2. The url points to github.com => checkout directory with `svn`
    #     3. Else simply use `wget` to download the `cls.url` content,
    #
    #     Overwrite if you need custom downloader
    #     """
    #     if self.rawdata_files_exist() and not force:
    #         self.__logger__.info("Dataset already exists. Skipping download.")
    #         return
    #
    #     if url is None:
    #         if self.base_url is None:
    #             self.__logger__.info("Dataset provides no url. Assumed offline")
    #             return
    #         url = self.base_url
    #
    #     self.__logger__.debug("STARTING TO DOWNLOAD DATASET.")
    #     self.download_from_url(str(url))
    #     self.__logger__.debug("FINISHED TO DOWNLOAD DATASET.")

    @classmethod
    def info(cls):
        r"""Open dataset information in browser."""
        if cls.info_url is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.info_url)

    def _repr_html_(self):
        if hasattr(self.dataset, "_repr_html_"):
            header = f"<h3>{self.__class__.__name__}</h3>"
            # noinspection PyProtectedMember
            html_repr = self.dataset._repr_html_()  # pylint: disable=protected-access
            return header + html_repr
        raise NotImplementedError


##########################################
# Dataset consisting of Single DataFrame #
##########################################


class SimpleDataset(BaseDataset):
    r"""Dataset class that consists of a singular DataFrame."""

    # def __getattr__(self, key):
    #     r"""Attribute lookup."""
    #     if key in self.dataset:
    #         return self.dataset[key]
    #     if hasattr(self.dataset, key):
    #         return getattr(self.dataset, key)
    #     raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")

    @cached_property
    def dataset_files(self) -> PathType:
        r"""Return the dataset files."""
        return self.__class__.__name__ + ".feather"

    @cached_property
    def dataset_paths(self) -> Path:
        r"""Path to raw data."""
        return self.dataset_dir / (self.dataset_files or "")

    @abstractmethod
    def _clean(self) -> None:
        r"""Clean the dataset."""

    @abstractmethod
    def _load(self) -> DATASET_OBJECT:
        r"""Load the dataset."""

    def _download(self) -> None:
        r"""Download the dataset."""
        assert self.base_url is not None
        self.download_from_url(self.base_url)

    @cached_property
    def dataset(self) -> DATASET_OBJECT:
        r"""Store cached version of dataset."""
        return self.load()

    def load(self) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        if not self.dataset_files_exist():
            self.clean()
        else:
            self.__logger__.debug("Dataset files already exist!")

        self.__logger__.debug("STARTING TO LOAD DATASET.")
        ds = self._load()
        self.__logger__.debug("FINISHED TO LOAD DATASET.")
        return ds

    def clean(self, *, force: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        if self.dataset_files_exist() and not force:
            self.__logger__.debug("Dataset files already exist, skipping.")
            return

        if not self.rawdata_files_exist():
            self.download()

        self.__logger__.debug("STARTING TO CLEAN DATASET.")
        self._clean()
        self.__logger__.debug("FINISHED TO CLEAN DATASET.")

    def download(self, *, force: bool = True) -> None:
        r"""Download the dataset."""
        if self.rawdata_files_exist() and not force:
            self.__logger__.info("Dataset already exists. Skipping download.")
            return

        if self.base_url is None:
            self.__logger__.info("Dataset provides no url. Assumed offline")
            return

        self.__logger__.debug("STARTING TO DOWNLOAD DATASET.")
        self._download()
        self.__logger__.debug("FINISHED TO DOWNLOAD DATASET.")


#############################################
# Dataset consisting of Multiple DataFrames #
#############################################


class Dataset(BaseDataset, Mapping, Generic[KeyType]):
    r"""Dataset class that consists of a multiple DataFrames.

    The Datasets are accessed by their index.
    We subclass Mapping to provide the mapping interface.
    """

    __slots__ = ()  # Do we need this?

    def __init__(self, *, initialize: bool = True, reset: bool = False):
        r"""Initialize the Dataset."""
        super().__init__(initialize=initialize, reset=reset)
        for key in self.index:
            if isinstance(key, str) and not hasattr(self, key):
                setattr(self, key, self[key])

    # def __getattr__(self, key):
    #     r"""Attribute lookup for index."""
    #     if self.index is not None and key in self.index:
    #         if self.dataset[key] is None:
    #             self.load(key=key)
    #         return self.dataset[key]
    #     raise AttributeError(f"No attribute {key} in {self.__class__.__name__}")

    def __repr__(self):
        r"""Pretty Print."""
        if len(self.index) > 6:
            indices = list(self.index)
            selection = [str(indices[k]) for k in [0, 1, 2, -2, -1]]
            selection[2] = "..."
            index_str = ", ".join(selection)
        else:
            index_str = repr(self.index)
        return f"{self.__class__.__name__}{index_str}"

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyType]:
        r"""Return the index of the dataset."""
        # implement loading of dataset

    @abstractmethod
    def _load(self, key: KeyType) -> Any:
        r"""Clean the selected DATASET_OBJECT."""

    @abstractmethod
    def _clean(self, key: KeyType) -> None:
        r"""Clean the selected DATASET_OBJECT."""

    @cached_property
    def dataset(self) -> MutableMapping[KeyType, DATASET_OBJECT]:
        r"""Store cached version of dataset."""
        return {key: None for key in self.index}

    @property
    def dataset_files(self) -> Mapping[KeyType, PathType]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.feather" for key in self.index}

    @cached_property
    def dataset_paths(self) -> Mapping[KeyType, Path]:
        r"""Absolute paths to the dataset files for each key."""
        return {key: self.dataset_dir / self.dataset_files[key] for key in self.index}

    def rawdata_files_exist(self, key: Optional[KeyType] = None) -> bool:
        r"""Check if raw data files exist."""
        if key is None:
            return paths_exists(self.rawdata_paths)
        if isinstance(self.rawdata_paths, Mapping):
            return paths_exists(self.rawdata_paths[key])
        return paths_exists(self.rawdata_paths)

    def dataset_files_exist(self, key: Optional[KeyType] = None) -> bool:
        r"""Check if dataset files exist."""
        if key is None:
            return paths_exists(self.dataset_paths)
        if isinstance(self.dataset_paths, Mapping):
            return paths_exists(self.dataset_paths[key])
        return paths_exists(self.dataset_paths)

    def clean(self, key: Optional[KeyType] = None, force: bool = False) -> None:
        r"""Clean the selected DATASET_OBJECT.

        Parameters
        ----------
        key: Optional[KeyType] = None
            The key of the dataset to clean. If None, clean all dataset.
        force: bool = False
            Force cleaning of dataset.
        """
        # TODO: Do we need this code block?
        if not self.rawdata_files_exist(key=key):
            self.__logger__.debug("%s: missing, fetching it now!", key)
            self.download(key=key, force=force)

        if (
            key in self.dataset_files
            and self.dataset_files_exist(key=key)
            and not force
        ):
            self.__logger__.debug("%s: already exists, skipping.", key)
            return

        if key is None:
            self.__logger__.debug("STARTING TO CLEAN DATASET.")
            for key_ in self.index:
                self.clean(key=key_, force=force)
            self.__logger__.debug("FINISHED TO CLEAN DATASET.")
            return

        self.__logger__.debug("%s: STARTING TO CLEAN DATASET.", key)
        self._clean(key=key)
        self.__logger__.debug("%s: FINISHED TO CLEAN DATASET.", key)

    @overload
    def load(
        self, *, key: None = None, force: bool = False, **kwargs: Any
    ) -> Mapping[KeyType, Any]:
        ...

    @overload
    def load(self, *, key: KeyType = None, force: bool = False, **kwargs: Any) -> Any:
        ...

    def load(
        self, *, key: Optional[KeyType] = None, force: bool = False, **kwargs: Any
    ) -> Union[Any, Mapping[KeyType, Any]]:
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
        if not self.dataset_files_exist(key=key):
            self.clean(key=key, force=force)

        if key is None:
            # Download full dataset
            self.__logger__.debug("STARTING TO LOAD DATASET.")
            ds = {k: self.load(key=k, force=force, **kwargs) for k in self.index}
            self.__logger__.debug("FINISHED TO LOAD DATASET.")
            return ds

        # download specific key
        if key in self.dataset and self.dataset[key] is not None and not force:
            self.__logger__.debug("%s: dataset already exists, skipping!", key)
            return self.dataset[key]

        self.__logger__.debug("%s: STARTING TO LOAD DATASET.", key)
        self.dataset[key] = self._load(key=key)
        self.__logger__.debug("%s: FINISHED TO LOAD DATASET.", key)
        return self.dataset[key]

    def _download(self, *, key: KeyType = None) -> None:
        r"""Download the selected DATASET_OBJECT."""
        assert self.base_url is not None, "base_url is not set!"

        files: Nested[Optional[str]]
        if isinstance(self.rawdata_files, Mapping):
            files = self.rawdata_files[key]  # type: ignore[assignment]
        else:
            files = self.rawdata_files  # type: ignore[assignment]

        files: Nested[Path] = prepend_path(files, Path(), keep_none=False)
        files: set[Path] = flatten_nested(files, kind=Path)

        for file in files:
            self.download_from_url(self.base_url + file.name)

    def download(
        self,
        *,
        key: Optional[KeyType] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Download the dataset.

        Parameters
        ----------
        key: Optional[KeyType] = None
        force: bool = False
            Force re-downloading of dataset.
        """
        if self.base_url is None:
            self.__logger__.debug("Dataset provides no base_url. Assumed offline")
            return

        if self.rawdata_files is None:
            self.__logger__.debug("Dataset needs no raw data files. Skipping.")
            return

        if not force and self.rawdata_files_exist(key=key):
            self.__logger__.debug(
                "%s: Rawdata files already exist, skipping.", str(key)
            )
            return

        if key is None:
            # Download full dataset
            self.__logger__.debug("STARTING TO DOWNLOAD DATASET.")
            if isinstance(self.rawdata_files, Mapping):
                for key_ in self.rawdata_files:
                    self.download(key=key_, force=force, **kwargs)
            else:
                self._download()
            self.__logger__.debug("FINISHED TO DOWNLOAD DATASET.")
            return

        # Download specific key
        self.__logger__.debug("%s: STARTING TO DOWNLOAD DATASET.", key)
        self._download(key=key)
        self.__logger__.debug("%s: FINISHED TO DOWNLOAD DATASET.", key)
