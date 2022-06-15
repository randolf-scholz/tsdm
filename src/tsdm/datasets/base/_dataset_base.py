r"""Base Classes for dataset."""

from __future__ import annotations

__all__ = [
    # Classes
    "BaseDataset",
    "BaseDatasetMetaClass",
    "SingleFrameDataset",
    "MultiFrameDataset",
]

import inspect
import logging
import os
import subprocess
import webbrowser
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, MutableMapping, Sequence
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Generic, Optional, Union, overload
from urllib.parse import urlparse

import pandas
from pandas import DataFrame, Series

from tsdm.config import DATASETDIR, RAWDATADIR
from tsdm.util import flatten_nested, paths_exists, prepend_path
from tsdm.util.remote import download
from tsdm.util.types import KeyType, Nested, PathType

DATASET_OBJECT = Union[Series, DataFrame]
r"""Type hint for pandas objects."""


class BaseDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if os.environ.get("GENERATING_DOCS", False):
            cls.RAWDATA_DIR = Path(f"~/.tsdm/rawdata/{cls.__name__}/")
            cls.DATASET_DIR = Path(f"~/.tsdm/datasets/{cls.__name__}/")
        else:
            cls.RAWDATA_DIR = RAWDATADIR / cls.__name__
            cls.DATASET_DIR = DATASETDIR / cls.__name__

        super().__init__(*args, **kwargs)


class BaseDataset(ABC, metaclass=BaseDatasetMetaClass):
    r"""Abstract base class that all dataset must subclass.

    Implements methods that are available for all dataset classes.
    """

    BASE_URL: Optional[str] = None
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL: Optional[str] = None
    r"""HTTP address containing additional information about the dataset."""
    RAWDATA_DIR: Path
    r"""Location where the raw data is stored."""
    DATASET_DIR: Path
    r"""Location where the pre-processed data is stored."""
    LOGGER: logging.Logger
    r"""Logger for the dataset."""

    def __init__(self, *, initialize: bool = True, reset: bool = False):
        r"""Initialize the dataset."""
        if not inspect.isabstract(self):
            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
            self.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        if reset:
            self.clean()
        if initialize:
            self.load()

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return self.dataset.__len__()

    def __getitem__(self, idx):
        r"""Return the sample at index `idx`."""
        return self.dataset.__getitem__(idx)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        r"""Return the sample at index `idx`."""
        self.dataset[key] = value

    def __delitem__(self, key: Hashable, /) -> None:
        r"""Return the sample at index `idx`."""
        self.dataset.__delitem__(key)

    def __iter__(self) -> Iterator:
        r"""Return an iterator over the dataset."""
        return self.dataset.__iter__()

    def __repr__(self):
        r"""Return a string representation of the dataset."""
        return f"{self.__class__.__name__}\n{self.dataset}"

    @classmethod
    def info(cls):
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    @cached_property
    @abstractmethod
    def dataset(self) -> MutableMapping:
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
        return prepend_path(self.rawdata_files, parent=self.RAWDATA_DIR)

    @cached_property
    def dataset_paths(self) -> Nested[Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return prepend_path(self.dataset_files, parent=self.RAWDATA_DIR)

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

    @classmethod
    def download_from_url(cls, url: str) -> None:
        r"""Download files from a URL."""
        cls.LOGGER.info("Obtaining files from %s", url)
        parsed_url = urlparse(url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                f"kaggle competitions download -p {cls.RAWDATA_DIR} -c {kaggle_name}",
                shell=True,
                check=True,
            )
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                f"svn export --force {url.replace('tree/main', 'trunk')} {cls.RAWDATA_DIR}",
                shell=True,
                check=True,
            )
        else:  # default parsing, including for UCI dataset
            fname = url.split("/")[-1]
            download(url, cls.RAWDATA_DIR / fname)

        cls.LOGGER.info("Finished importing files from %s", url)


class FrameDataset(BaseDataset, ABC):
    r"""Base class for datasets that are stored as pandas.DataFrame."""

    default_format: str = "parquet"
    r"""Default format for the dataset."""

    @staticmethod
    def serialize(frame: DATASET_OBJECT, path: Path, /, **kwargs: Any) -> None:
        r"""Serialize the dataset."""
        file_type = path.suffix
        assert file_type.startswith("."), "File must have a suffix!"
        file_type = file_type[1:]
        if hasattr(frame, f"to_{file_type}"):
            pandas_writer = getattr(frame, f"read_{file_type}")
            pandas_writer(path, **kwargs)

        raise NotImplementedError(f"No loader for {file_type=}")

    @staticmethod
    def deserialize(path: Path, /, *, squeeze: bool = True) -> DATASET_OBJECT:
        r"""Deserialize the dataset."""
        file_type = path.suffix
        assert file_type.startswith("."), "File must have a suffix!"
        file_type = file_type[1:]

        if hasattr(pandas, f"read_{file_type}"):
            pandas_loader = getattr(pandas, f"read_{file_type}")
            pandas_object = pandas_loader(path)
            return pandas_object.squeeze() if squeeze else pandas_object

        raise NotImplementedError(f"No loader for {file_type=}")


class SingleFrameDataset(FrameDataset):
    r"""Dataset class that consists of a singular DataFrame."""

    def _repr_html_(self):
        if hasattr(self.dataset, "_repr_html_"):
            header = f"<h3>{self.__class__.__name__}</h3>"
            # noinspection PyProtectedMember
            html_repr = self.dataset._repr_html_()  # pylint: disable=protected-access
            return header + html_repr
        raise NotImplementedError

    @cached_property
    def dataset(self) -> DATASET_OBJECT:
        r"""Store cached version of dataset."""
        return self.load()

    @cached_property
    def dataset_files(self) -> PathType:
        r"""Return the dataset files."""
        return self.__class__.__name__ + f".{self.default_format}"

    @cached_property
    def dataset_paths(self) -> Path:
        r"""Path to raw data."""
        return self.DATASET_DIR / (self.dataset_files or "")

    @abstractmethod
    def _clean(self) -> DATASET_OBJECT | None:
        r"""Clean the dataset."""

    def _load(self) -> DATASET_OBJECT:
        r"""Load the dataset."""
        return self.deserialize(self.dataset_paths)

    def _download(self) -> None:
        r"""Download the dataset."""
        assert self.BASE_URL is not None, "base_url is not set!"

        files: Nested[Path] = prepend_path(self.rawdata_files, Path(), keep_none=False)
        files: set[Path] = flatten_nested(files, kind=Path)

        for file in files:
            self.download_from_url(self.BASE_URL + file.name)

    def load(self) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        if not self.dataset_files_exist():
            self.clean()
        else:
            self.LOGGER.debug("Dataset files already exist!")

        self.LOGGER.debug("STARTING TO LOAD DATASET.")
        ds = self._load()
        self.LOGGER.debug("FINISHED TO LOAD DATASET.")
        return ds

    def clean(self, *, force: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        if self.dataset_files_exist() and not force:
            self.LOGGER.debug("Dataset files already exist, skipping.")
            return

        if not self.rawdata_files_exist():
            self.download()

        self.LOGGER.debug("STARTING TO CLEAN DATASET.")
        df = self._clean()
        if df is not None:
            self.serialize(df, self.dataset_paths)
        self.LOGGER.debug("FINISHED TO CLEAN DATASET.")

    def download(self, *, force: bool = True) -> None:
        r"""Download the dataset."""
        if self.rawdata_files_exist() and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        if self.BASE_URL is None:
            self.LOGGER.info("Dataset provides no url. Assumed offline")
            return

        self.LOGGER.debug("STARTING TO DOWNLOAD DATASET.")
        self._download()
        self.LOGGER.debug("FINISHED TO DOWNLOAD DATASET.")


class MultiFrameDataset(FrameDataset, Mapping, Generic[KeyType]):
    r"""Dataset class that consists of a multiple DataFrames.

    The Datasets are accessed by their index.
    We subclass Mapping to provide the mapping interface.
    """

    def __init__(self, *, initialize: bool = True, reset: bool = False):
        r"""Initialize the Dataset."""
        self.LOGGER.info("Adding keys as attributes.")
        for key in self.index:
            if isinstance(key, str) and not hasattr(self, key):
                _get_dataset = partial(self.__class__.load, key=key)
                _get_dataset.__doc__ = f"Load dataset for {key=}."
                setattr(self.__class__, key, property(_get_dataset))

        super().__init__(initialize=initialize, reset=reset)

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

    @cached_property
    def dataset(self) -> MutableMapping[KeyType, DATASET_OBJECT]:
        r"""Store cached version of dataset."""
        return {key: None for key in self.index}

    @cached_property
    def dataset_files(self) -> Mapping[KeyType, PathType]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.{self.default_format}" for key in self.index}

    @cached_property
    def dataset_paths(self) -> Mapping[KeyType, Path]:
        r"""Absolute paths to the dataset files for each key."""
        return {key: self.DATASET_DIR / self.dataset_files[key] for key in self.index}

    def _load(self, key: KeyType) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        return self.deserialize(self.dataset_paths[key])

    @abstractmethod
    def _clean(self, key: KeyType) -> DATASET_OBJECT | None:
        r"""Clean the selected DATASET_OBJECT."""

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
            self.LOGGER.debug("Raw files missing, fetching it now! <%s>", key)
            self.download(key=key, force=force)

        if (
            key in self.dataset_files
            and self.dataset_files_exist(key=key)
            and not force
        ):
            self.LOGGER.debug("Clean files already exists, skipping <%s>", key)
            return

        if key is None:
            self.LOGGER.debug("Starting to clean dataset.")
            for key_ in self.index:
                self.clean(key=key_, force=force)
            self.LOGGER.debug("Finished cleaning dataset.")
            return

        self.LOGGER.debug("Starting to clean dataset <%s>", key)
        df = self._clean(key=key)
        if df is not None:
            self.serialize(df, self.dataset_paths[key])
        self.LOGGER.debug("Finished cleaning dataset <%s>", key)

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
            self.LOGGER.debug("Starting to load  dataset.")
            ds = {k: self.load(key=k, force=force, **kwargs) for k in self.index}
            self.LOGGER.debug("Finished loading  dataset.")
            return ds

        # download specific key
        if key in self.dataset and self.dataset[key] is not None and not force:
            self.LOGGER.debug("Dataset already exists, skipping! <%s>", key)
            return self.dataset[key]

        self.LOGGER.debug("Starting to load  dataset <%s>", key)
        self.dataset[key] = self._load(key=key)
        self.LOGGER.debug("Finished loading  dataset <%s>", key)
        return self.dataset[key]

    def _download(self, *, key: KeyType = None) -> None:
        r"""Download the selected DATASET_OBJECT."""
        assert self.BASE_URL is not None, "base_url is not set!"

        files: Nested[Optional[str]]
        if isinstance(self.rawdata_files, Mapping):
            files = self.rawdata_files[key]  # type: ignore[assignment]
        else:
            files = self.rawdata_files  # type: ignore[assignment]

        files: Nested[Path] = prepend_path(files, Path(), keep_none=False)
        files: set[Path] = flatten_nested(files, kind=Path)

        for file in files:
            self.download_from_url(self.BASE_URL + file.name)

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
        if self.BASE_URL is None:
            self.LOGGER.debug("Dataset provides no base_url. Assumed offline")
            return

        if self.rawdata_files is None:
            self.LOGGER.debug("Dataset needs no raw data files. Skipping.")
            return

        if not force and self.rawdata_files_exist(key=key):
            self.LOGGER.debug("Rawdata files already exist, skipping. <%s>", str(key))
            return

        if key is None:
            # Download full dataset
            self.LOGGER.debug("Starting to download dataset.")
            if isinstance(self.rawdata_files, Mapping):
                for key_ in self.rawdata_files:
                    self.download(key=key_, force=force, **kwargs)
            else:
                self._download()
            self.LOGGER.debug("Finished downloading dataset.")
            return

        # Download specific key
        self.LOGGER.debug("Starting to download dataset <%s>", key)
        self._download(key=key)
        self.LOGGER.debug("Finished downloading dataset <%s>", key)
