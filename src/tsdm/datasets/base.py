r"""Base Classes for dataset."""
# NOTE: signature of metaclass.__init__ should match that of type.__new__
# NOTE: type.__new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any)
# NOTE: type.__init__(self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any)

__all__ = [
    # Classes
    "BaseDataset",
    "BaseDatasetMetaClass",
    "SingleFrameDataset",
    "MultiFrameDataset",
    # Types
    "DATASET_OBJECT",
]

import inspect
import logging
import os
import subprocess
import warnings
import webbrowser
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from functools import cached_property, partial
from pathlib import Path
from typing import Any, ClassVar, Generic, Optional, TypeAlias, overload
from urllib.parse import urlparse

import pandas
from pandas import DataFrame, Series

from tsdm.config import CONFIG
from tsdm.types.aliases import Nested, PathLike, Schema
from tsdm.types.variables import StringVar as Key
from tsdm.utils import flatten_nested, paths_exists, prepend_path
from tsdm.utils.hash import validate_file_hash, validate_table_hash
from tsdm.utils.remote import download
from tsdm.utils.strings import repr_mapping

DATASET_OBJECT: TypeAlias = Series | DataFrame
r"""Type hint for pandas objects."""


class BaseDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if "RAWDATA_DIR" not in namespace:
            if os.environ.get("GENERATING_DOCS", False):
                cls.RAWDATA_DIR = Path("~/.tsdm/rawdata/") / cls.__name__
            else:
                cls.RAWDATA_DIR = CONFIG.RAWDATADIR / cls.__name__

        if "DATASET_DIR" not in namespace:
            if os.environ.get("GENERATING_DOCS", False):
                cls.DATASET_DIR = Path("~/.tsdm/datasets") / cls.__name__
            else:
                cls.DATASET_DIR = CONFIG.DATASETDIR / cls.__name__

        # print(f"Setting Attribute {cls}.RAWDATA_DIR = {cls.RAWDATA_DIR}")
        # print(f"{cls=}\n\n{args=}\n\n{kwargs.keys()=}\n\n")

    # def __getitem__(cls, parent: type[BaseDataset]) -> type[BaseDataset]:
    #     # if inspect.isabstract(cls):
    #     cls.RAWDATA_DIR = parent.RAWDATA_DIR
    #     print(f"Setting {cls}.RAWDATA_DIR = {parent.RAWDATA_DIR=}")
    #     return cls
    # return super().__getitem__(parent)


class BaseDataset(Generic[Key], ABC, metaclass=BaseDatasetMetaClass):
    r"""Abstract base class that all dataset must subclass.

    Implements methods that are available for all dataset classes.


    We allow for systematic validation of the datasets.
    Users can check the validity of the datasets by implementing the following attributes/properites:

    - rawdata_hashes: Used to verify rawdata files after `download` and before `clean`.
    - dataset_hashes: Used to verify cleaned dataset files after `clean` and before `load`.
    - table_hashes: Used to verify in-memory tables after `load`.
    - table_schemas: Used to verify in-memory tables after `load`.

    """

    BASE_URL: ClassVar[Optional[str]] = None
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL: ClassVar[Optional[str]] = None
    r"""HTTP address containing additional information about the dataset."""
    RAWDATA_DIR: ClassVar[Path]
    r"""Location where the raw data is stored."""
    DATASET_DIR: ClassVar[Path]
    r"""Location where the pre-processed data is stored."""
    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the dataset."""

    # instance attribute
    __version__: str
    r"""Version of the dataset, keep empty for unversioned dataset."""

    # Validation - Implement on per dataset basis!
    rawdata_hashes: Mapping[str, str] = NotImplemented
    r"""Hashes of the raw dataset file(s)."""
    dataset_hashes: Mapping[Key, str] = NotImplemented
    r"""Hashes of the cleaned dataset file(s)."""
    table_hashes: Mapping[Key, str] = NotImplemented
    r"""Hashes of the in-memory cleaned dataset table(s)."""
    table_schemas: Mapping[Key, Schema] = NotImplemented
    r"""Schema of the in-memory cleaned dataset table(s)."""

    def __init__(
        self,
        *,
        initialize: bool = True,
        reset: bool = False,
        version: str = "",
    ) -> None:
        r"""Initialize the dataset."""
        self.__version__ = version
        self.RAWDATA_DIR = self.RAWDATA_DIR / self.__version__  # type: ignore[misc]
        self.DATASET_DIR = self.DATASET_DIR / self.__version__  # type: ignore[misc]

        if not inspect.isabstract(self):
            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
            self.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        if reset:
            self.clean()
        if initialize:
            self.load()

    def __repr__(self) -> str:
        r"""Return a string representation of the dataset."""
        return f"{self.__class__.__name__}\n{self.dataset}"

    @classmethod
    def info(cls) -> None:
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    @cached_property
    @abstractmethod
    def dataset(self) -> Any | MutableMapping:
        r"""Store cached version of dataset."""
        return self.load()

    @property
    @abstractmethod
    def dataset_files(self) -> Mapping[Key, PathLike]:
        r"""Relative paths to the cleaned dataset file(s)."""

    @property
    @abstractmethod
    def rawdata_files(self) -> Sequence[PathLike]:
        r"""Relative paths to the raw dataset file(s)."""

    @cached_property
    def dataset_paths(self) -> Mapping[Key, Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return prepend_path(self.dataset_files, parent=self.RAWDATA_DIR)

    @cached_property
    def rawdata_paths(self) -> Mapping[Key, list[Path]]:
        r"""Absolute paths to the raw dataset file(s)."""
        if self.rawdata_mapping is not NotImplemented:
            return prepend_path(self.rawdata_mapping, parent=self.RAWDATA_DIR)
        else:
            return {None: prepend_path(self.rawdata_files, parent=self.RAWDATA_DIR)}

    def rawdata_files_exist(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.rawdata_paths)

    def dataset_files_exist(self) -> bool:
        r"""Check if dataset files exist."""
        return paths_exists(self.dataset_paths)

    @abstractmethod
    def clean(self) -> None:
        r"""Clean an already downloaded raw dataset and stores it in self.data_dir.

        Preferably, use the '.parquet' data format.
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
        cls.LOGGER.info("Downloading from %s", url)
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


# class VersionedDataset(BaseDataset, ABC):
#     """Base class for datasets that have multiple versions."""
#
#     VERSION: str
#     """Version of the dataset."""
#
#     def __init__(
#         self, *, version: str, initialize: bool = True, reset: bool = False
#     ) -> None:
#         r"""Initialize the dataset."""
#         self.VERSION = version
#         self.RAWDATA_DIR = self.RAWDATA_DIR / self.VERSION
#         self.DATASET_DIR = self.DATASET_DIR / self.VERSION
#         super().__init__(initialize=initialize, reset=reset)
#
#     def __repr__(self) -> str:
#         r"""Return a string representation of the dataset."""
#         return f"{self.__class__.__name__}@{self.VERSION}\n{self.dataset}"


class FrameDataset(BaseDataset, ABC):
    r"""Base class for datasets that are stored as pandas.DataFrame."""

    DEFAULT_FILE_FORMAT: ClassVar[str] = "parquet"
    r"""Default format for the dataset."""

    @staticmethod
    def serialize(frame: DATASET_OBJECT, path: Path, /, **kwargs: Any) -> None:
        r"""Serialize the dataset."""
        file_type = path.suffix
        assert file_type.startswith("."), "File must have a suffix!"
        file_type = file_type[1:]

        if isinstance(frame, Series):
            frame = frame.to_frame()

        if hasattr(frame, f"to_{file_type}"):
            pandas_writer = getattr(frame, f"to_{file_type}")
            pandas_writer(path, **kwargs)
            return

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

    __dataset: DATASET_OBJECT = NotImplemented
    """INTERNAL: the dataset."""

    def _repr_html_(self) -> str:
        if hasattr(self.dataset, "_repr_html_"):
            header = f"<h3>{self.__class__.__name__}</h3>"
            # noinspection PyProtectedMember
            html_repr = self.dataset._repr_html_()  # pylint: disable=protected-access
            return header + html_repr
        raise NotImplementedError

    @cached_property
    def dataset(self) -> DATASET_OBJECT:
        r"""Store cached version of dataset."""
        if self.__dataset is NotImplemented:
            self.load()
        return self.__dataset

    @cached_property
    def dataset_files(self) -> PathLike:
        r"""Return the dataset files."""
        return self.__class__.__name__ + f".{self.DEFAULT_FILE_FORMAT}"

    @cached_property
    def dataset_paths(self) -> Path:
        r"""Path to raw data."""
        return self.DATASET_DIR / (self.dataset_files or "")

    @abstractmethod
    def clean_table(self) -> DATASET_OBJECT | None:
        r"""Clean the dataset."""

    def load_table(self) -> DATASET_OBJECT:
        r"""Load the dataset."""
        return self.deserialize(self.dataset_paths)

    def download_all_files(self) -> None:
        r"""Download the dataset."""
        if self.BASE_URL is None:
            raise ValueError("No base URL provided!")

        nested_files: Nested[Path] = prepend_path(
            self.rawdata_files, Path(), keep_none=False
        )
        files: set[Path] = flatten_nested(nested_files, kind=Path)

        for file in files:
            self.download_from_url(self.BASE_URL + file.name)

    def load(self, *, force: bool = True, validate: bool = True) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        if not self.dataset_files_exist():
            self.clean(force=force, validate=validate)
        else:
            self.LOGGER.debug("Dataset files already exist!")

        self.LOGGER.debug("Starting to load dataset.")
        table = self.load_table()
        table.name = self.__class__.__name__
        self.__dataset = table
        self.LOGGER.debug("Finished loading dataset.")

        if validate:  # check the in-memory table hash
            validate_table_hash(table, reference=self.table_hashes)

        return table

    def clean(self, *, force: bool = True, validate: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        if self.dataset_files_exist() and not force:
            self.LOGGER.debug("Dataset files already exist, skipping.")
            return

        if not self.rawdata_files_exist():
            self.download(force=force, validate=validate)

        if validate:  # check the raw file hashes
            validate_file_hash(self.rawdata_paths, reference=self.rawdata_hashes)

        self.LOGGER.debug("Starting to clean dataset.")
        df = self.clean_table()
        if df is not None:
            self.LOGGER.info("Serializing dataset.")
            self.serialize(df, self.dataset_paths)
        self.LOGGER.debug("Finished cleaning dataset.")

        if validate:  # check the cleaned file hashes
            validate_file_hash(self.dataset_paths, reference=self.dataset_hashes)

    def download(self, *, force: bool = True, validate: bool = True) -> None:
        r"""Download the dataset."""
        if self.rawdata_files_exist() and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        self.LOGGER.debug("Starting to download dataset.")
        self.download_all_files()
        self.LOGGER.debug("Starting downloading dataset.")

        if validate:  # check the raw file hashes
            validate_file_hash(self.rawdata_paths, reference=self.rawdata_hashes)


class MultiFrameDataset(FrameDataset, Generic[Key]):
    r"""Dataset class that consists of multiple tables.

    The tables are stored in a dictionary-like object.
    """

    def __init__(self, *, initialize: bool = True, reset: bool = False) -> None:
        r"""Initialize the Dataset."""
        self.LOGGER.info("Adding keys as attributes.")
        while initialize:
            non_string_keys = {key for key in self.KEYS if not isinstance(key, str)}
            if non_string_keys:
                warnings.warn(
                    f"Not adding keys as attributes! "
                    f"Keys {non_string_keys!r} are not strings!",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            key_attributes = {
                key for key in self.KEYS if isinstance(key, str) and hasattr(self, key)
            }
            if key_attributes:
                warnings.warn(
                    f"Not adding keys as attributes! "
                    f"Keys {key_attributes!r} already exist as attributes!",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            for key in self.KEYS:
                assert isinstance(key, str) and not hasattr(self, key)
                _get_table = partial(self.__class__.load, key=key)
                _get_table.__doc__ = f"Load dataset for {key=}."
                setattr(self.__class__, key, property(_get_table))
            break

        super().__init__(initialize=initialize, reset=reset)

    # def __len__(self) -> int:
    #     r"""Return the number of samples in the dataset."""
    #     return self.dataset.__len__()
    #
    # def __getitem__(self, idx):
    #     r"""Return the sample at index `idx`."""
    #     return self.dataset.__getitem__(idx)
    #
    # def __iter__(self) -> Iterator:
    #     r"""Return an iterator over the dataset."""
    #     return self.dataset.__iter__()

    def __repr__(self) -> str:
        r"""Pretty Print."""
        return repr_mapping(self.dataset, title=self.__class__.__name__)

    @property
    @abstractmethod
    def KEYS(self) -> Sequence[Key]:
        r"""Return the index of the dataset."""
        # TODO: use abstract-attribute!
        # https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

    @cached_property
    def dataset(self) -> MutableMapping[Key, DATASET_OBJECT]:
        r"""Store cached version of dataset."""
        return {key: None for key in self.KEYS}

    @cached_property
    def dataset_files(self) -> Mapping[Key, str]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.{self.DEFAULT_FILE_FORMAT}" for key in self.KEYS}

    @cached_property
    def dataset_paths(self) -> dict[Key, Path]:
        r"""Absolute paths to the dataset files for each key."""
        return {
            key: self.DATASET_DIR / file for key, file in self.dataset_files.items()
        }

    def rawdata_files_exist(self, key: Optional[Key] = None) -> bool:
        r"""Check if raw data files exist."""
        if isinstance(self.rawdata_paths, Mapping) and key is not None:
            return paths_exists(self.rawdata_paths[key])
        return super().rawdata_files_exist()

    def dataset_files_exist(self, key: Optional[Key] = None) -> bool:
        r"""Check if dataset files exist."""
        if isinstance(self.dataset_files, Mapping) and key is not None:
            return paths_exists(self.dataset_files[key])
        return super().dataset_files_exist()

    @abstractmethod
    def clean_table(self, key: Key) -> DATASET_OBJECT | None:
        r"""Create thecleaned  table for the given key."""

    def clean(
        self,
        key: Optional[Key] = None,
        *,
        force: bool = False,
        validate: bool = True,
    ) -> None:
        r"""Clean the selected DATASET_OBJECT.

        Args:
            key: The key of the dataset to clean. If None, clean all dataset.
            force: Force cleaning of dataset.
            validate: Validate the dataset after cleaning.
        """
        # TODO: Do we need this code block?
        if not self.rawdata_files_exist(key=key):
            self.LOGGER.debug("Raw files missing, fetching it now! <%s>", key)
            self.download(key=key, force=force, validate=validate)

        if (
            key in self.dataset_files
            and self.dataset_files_exist(key=key)
            and not force
        ):
            self.LOGGER.debug("Clean files already exists, skipping <%s>", key)
            assert key is not None
            if validate:
                validate_file_hash(
                    self.dataset_paths[key], reference=self.dataset_hashes[key]
                )
            return

        if key is None:
            self.LOGGER.debug("Starting to clean dataset.")
            for key_ in self.KEYS:
                self.clean(key=key_, force=force, validate=validate)
            self.LOGGER.debug("Finished cleaning dataset.")
            return

        self.LOGGER.debug("Starting to clean dataset <%s>", key)

        df = self.clean_table(key=key)
        if df is not None:
            self.LOGGER.info("Serializing dataset <%s>", key)
            self.serialize(df, self.dataset_paths[key])
        self.LOGGER.debug("Finished cleaning dataset <%s>", key)

        if validate:
            validate_file_hash(
                self.dataset_paths[key], reference=self.dataset_hashes[key]
            )

    def load_table(self, key: Key) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        return self.deserialize(self.dataset_paths[key])

    @overload
    def load(
        self,
        *,
        key: None = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Mapping[Key, Any]:
        ...

    @overload
    def load(
        self,
        *,
        key: Optional[Key] = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        ...

    def load(
        self,
        *,
        key: Optional[Key] = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Mapping[Key, DATASET_OBJECT] | DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT.

        Args:
            key: The key associated with the datset
            force: Reload the dataset even if it already exists.
            validate: Validate the dataset against hash.

        Returns:
            The loaded dataset
        """
        if not self.dataset_files_exist(key=key):
            self.clean(key=key, force=force)

        if key is None:
            # Download full dataset
            self.LOGGER.debug("Starting to load  dataset.")
            ds = {
                k: self.load(key=k, force=force, validate=validate, **kwargs)
                for k in self.KEYS
            }
            self.LOGGER.debug("Finished loading  dataset.")
            return ds

        # download specific key
        if key in self.dataset and self.dataset[key] is not None and not force:
            self.LOGGER.debug("Dataset already exists, skipping! <%s>", key)
            return self.dataset[key]

        self.LOGGER.debug("Starting to load  dataset <%s>", key)
        table = self.load_table(key=key)
        table.name = key
        self.dataset[key] = table
        self.LOGGER.debug("Finished loading  dataset <%s>", key)

        if validate:
            validate_table_hash(table, reference=self.table_hashes[key])

        return self.dataset[key]

    def download_table(self, *, key: Key = NotImplemented) -> None:
        r"""Download the selected DATASET_OBJECT."""
        assert self.BASE_URL is not None, "base_url is not set!"

        rawdata_files: Nested[Optional[PathLike]]
        if isinstance(self.rawdata_files, Mapping):
            assert key is not NotImplemented, "key must be provided!"
            rawdata_files = self.rawdata_files[key]
        else:
            rawdata_files = self.rawdata_files

        nested_files: Nested[Path] = prepend_path(
            rawdata_files, Path(), keep_none=False
        )
        files: set[Path] = flatten_nested(nested_files, kind=Path)

        for file in files:
            self.download_from_url(self.BASE_URL + file.name)

    def download(
        self,
        key: Optional[Key] = None,
        *,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Download the dataset.

        Args:
            key: The key of the rawdata file
            validate: Validate the downloaded files.
            force: Force re-downloading of dataset.
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
                    self.download(key=key_, force=force, validate=validate, **kwargs)
            else:
                self.download_table()
            self.LOGGER.debug("Finished downloading dataset.")

            if validate:
                validate_file_hash(self.rawdata_paths, reference=self.rawdata_hashes)
            return

        # Download specific key
        self.LOGGER.debug("Starting to download dataset <%s>", key)
        self.download_table(key=key)
        self.LOGGER.debug("Finished downloading dataset <%s>", key)
