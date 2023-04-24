r"""Base Classes for dataset."""
# NOTE: signature of metaclass.__init__ should match that of type.__new__
# NOTE: type.__new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any)
# NOTE: type.__init__(self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any)

__all__ = [
    # Protocol
    "Dataset",
    # Classes
    "BaseDataset",
    "BaseDatasetMetaClass",
    "SingleTableDataset",
    "MultiTableDataset",
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
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from functools import cached_property, partial
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)
from urllib.parse import urlparse

import pandas
from pandas import DataFrame, Series

from tsdm.config import CONFIG
from tsdm.types.aliases import PathLike, Schema
from tsdm.types.variables import str_var as Key
from tsdm.utils import paths_exists
from tsdm.utils.hash import validate_file_hash, validate_table_hash
from tsdm.utils.lazydict import LazyDict
from tsdm.utils.remote import download
from tsdm.utils.strings import repr_mapping

DATASET_OBJECT: TypeAlias = Series | DataFrame
r"""Type hint for pandas objects."""


@runtime_checkable
class Dataset(Protocol):
    """Protocol for Dataset."""

    INFO_URL: ClassVar[Optional[str]] = None
    r"""HTTP address containing additional information about the dataset."""
    RAWDATA_DIR: ClassVar[Path]
    r"""The directory where the raw data is stored."""
    DATASET_DIR: ClassVar[Path]
    r"""The directory where the dataset is stored."""

    @classmethod
    def info(cls) -> None:
        """Print information about the dataset."""

    @property
    def rawdata_paths(self) -> Sequence[Path]:
        """Return list of paths to the rawdata files."""

    def clean(self) -> None:
        """Clean the dataset."""

    def download(self) -> None:
        """Download the dataset."""

    def load(self) -> None:
        """Load the dataset."""


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
    DEFAULT_FILE_FORMAT: ClassVar[str] = "parquet"
    r"""Default format for the dataset."""

    # instance attribute
    __version__: str
    r"""Version of the dataset, keep empty for unversioned dataset."""
    rawdata_hashes: Mapping[str, str] = NotImplemented
    r"""Hashes of the raw dataset file(s)."""
    rawdata_schemas: Mapping[str, Mapping[str, str]] = NotImplemented
    r"""Schemas for the raw dataset file(s)."""

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
        return f"{self.__class__.__name__}()"

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

    @classmethod
    def info(cls) -> None:
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    @property
    @abstractmethod
    def rawdata_files(self) -> Sequence[PathLike]:
        r"""Relative paths to the raw dataset file(s)."""

    @cached_property
    def rawdata_paths(self) -> Mapping[str, Path]:
        r"""Absolute paths corresponding to the raw dataset file(s)."""
        return {
            str(fname): (self.RAWDATA_DIR / fname).absolute()
            for fname in self.rawdata_files
        }

    def rawdata_files_exist(self, key: Optional[str] = None) -> bool:
        r"""Check if raw data files exist."""
        if key is None:
            return paths_exists(self.rawdata_paths)
        if key not in self.rawdata_paths:
            raise KeyError(f"{key=} not in {self.rawdata_paths=}")
        return paths_exists(key)

    @abstractmethod
    def clean(self) -> None:
        r"""Clean an already downloaded raw dataset and stores it in self.data_dir.

        Preferably, use the '.parquet' data format.
        """

    @abstractmethod
    def load(self):
        r"""Load the pre-processed dataset."""

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

    def download_file(self, fname: str) -> None:
        r"""Download a single file."""
        if self.BASE_URL is None:
            self.LOGGER.debug("Dataset provides no base_url. Assumed offline")
            return

        self.download_from_url(self.BASE_URL + fname)

    def download(
        self, *, key: Optional[str] = None, force: bool = True, validate: bool = True
    ) -> None:
        r"""Download the dataset."""
        # Recurse if key is None.
        if key is None:
            self.LOGGER.debug("Starting to download dataset.")
            for _key in self.rawdata_paths:
                self.download(key=_key, force=force, validate=validate)
            self.LOGGER.debug("Finished downloading dataset.")
            return

        # Check if file already exists.
        if self.rawdata_files_exist(key) and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        # Check if dataset provides a base_url.
        if self.BASE_URL is None:
            self.LOGGER.debug("Dataset provides no base_url. Assumed offline")
            return

        # Download file.
        self.LOGGER.debug("Starting to download dataset <%s>", key)
        self.download_file(key)
        self.LOGGER.debug("Finished downloading dataset <%s>", key)

        # Validate file.
        if validate and self.rawdata_hashes is not NotImplemented:
            validate_file_hash(self.rawdata_paths[key], reference=self.rawdata_hashes)


class SingleTableDataset(BaseDataset):
    r"""Dataset class that consists of a singular DataFrame."""

    __dataset: DATASET_OBJECT = NotImplemented
    """INTERNAL: the dataset."""

    # Validation - Implement on per dataset basis!

    dataset_hash: str = NotImplemented
    r"""Hashes of the cleaned dataset file(s)."""
    table_hash: str = NotImplemented
    r"""Hashes of the in-memory cleaned dataset table(s)."""
    table_schema: Schema = NotImplemented
    r"""Schema of the in-memory cleaned dataset table(s)."""

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
    def dataset_file(self) -> PathLike:
        r"""Return the dataset files."""
        return f"{self.__class__.__name__}.{self.DEFAULT_FILE_FORMAT}"

    @cached_property
    def dataset_path(self) -> Path:
        r"""Path to raw data."""
        return self.DATASET_DIR / self.dataset_file

    def dataset_file_exists(self) -> bool:
        r"""Check if dataset file exists."""
        return paths_exists(self.dataset_path)

    @abstractmethod
    def clean_table(self) -> DATASET_OBJECT | None:
        r"""Generate the cleaned dataset table.

        If table is returned, the `self.serialize` method is used to write it to disk.
        If manually writing the table to disk, return None.
        """

    def load_table(self) -> DATASET_OBJECT:
        r"""Load the dataset.

        By default, `self.deserialize` is used to load the table from disk.
        Override this method if you want to customize loading the table from disk.
        """
        return self.deserialize(self.dataset_path)

    def load(self, *, force: bool = True, validate: bool = True) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        if not self.dataset_file_exists():
            self.clean(force=force, validate=validate)
        else:
            self.LOGGER.debug("Dataset files already exist!")

        if validate and self.dataset_hash is not NotImplemented:
            validate_file_hash(self.dataset_path, reference=self.dataset_hash)

        self.LOGGER.debug("Starting to load dataset.")
        table = self.load_table()
        table.name = self.__class__.__name__
        self.__dataset = table
        self.LOGGER.debug("Finished loading dataset.")

        if validate and self.table_hash is not NotImplemented:
            validate_table_hash(table, reference=self.table_hash)

        return table

    def clean(self, *, force: bool = True, validate: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        if self.dataset_file_exists() and not force:
            self.LOGGER.debug("Dataset files already exist, skipping.")
            return

        if not self.rawdata_files_exist():
            self.download(force=force, validate=validate)

        if validate and self.rawdata_hashes is not NotImplemented:
            validate_file_hash(self.rawdata_paths, reference=self.rawdata_hashes)

        self.LOGGER.debug("Starting to clean dataset.")
        df = self.clean_table()
        if df is not None:
            self.LOGGER.info("Serializing dataset.")
            self.serialize(df, self.dataset_path)
        self.LOGGER.debug("Finished cleaning dataset.")

        if validate and self.dataset_hash is not NotImplemented:
            validate_file_hash(self.dataset_path, reference=self.dataset_hash)


class MultiTableDataset(BaseDataset, Mapping[Key, DATASET_OBJECT]):
    r"""Dataset class that consists of multiple tables.

    The tables are stored in a dictionary-like object.
    """

    # Validation - Implement on per dataset basis!
    dataset_hashes: Mapping[Key, str] = NotImplemented
    r"""Hashes of the cleaned dataset file(s)."""
    table_hashes: Mapping[Key, str] = NotImplemented
    r"""Hashes of the in-memory cleaned dataset table(s)."""
    table_schemas: Mapping[Key, Schema] = NotImplemented
    r"""Schema of the in-memory cleaned dataset table(s)."""

    def __init__(self, *, initialize: bool = True, reset: bool = False) -> None:
        r"""Initialize the Dataset."""
        self.LOGGER.info("Adding keys as attributes.")
        while initialize:
            non_string_keys = {
                key for key in self.table_names if not isinstance(key, str)
            }
            if non_string_keys:
                warnings.warn(
                    f"Not adding keys as attributes! "
                    f"Keys {non_string_keys!r} are not strings!",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            key_attributes = {
                key
                for key in self.table_names
                if isinstance(key, str) and hasattr(self, key)
            }
            if key_attributes:
                warnings.warn(
                    f"Not adding keys as attributes! "
                    f"Keys {key_attributes!r} already exist as attributes!",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            for key in self.table_names:
                assert isinstance(key, str) and not hasattr(self, key)
                _get_table = partial(self.__class__.load, key=key)
                _get_table.__doc__ = f"Load dataset for {key=}."
                setattr(self.__class__, key, property(_get_table))
            break

        super().__init__(initialize=initialize, reset=reset)

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return self.tables.__len__()

    def __getitem__(self, idx: Key) -> DATASET_OBJECT:
        r"""Return the sample at index `idx`."""
        return self.tables.__getitem__(idx)

    def __iter__(self) -> Iterator[Key]:
        r"""Return an iterator over the dataset."""
        return self.tables.__iter__()

    def __repr__(self) -> str:
        r"""Pretty Print."""
        return repr_mapping(self, title=self.__class__.__name__)

    @property
    @abstractmethod
    def table_names(self) -> Sequence[Key]:
        r"""Return the index of the dataset."""
        # TODO: use abstract-attribute!
        # https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

    @cached_property
    def tables(self) -> MutableMapping[Key, DATASET_OBJECT]:
        r"""Store cached version of dataset."""
        return LazyDict({key: self.load_table for key in self.table_names})

    @cached_property
    def dataset_files(self) -> Mapping[Key, str]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.{self.DEFAULT_FILE_FORMAT}" for key in self.table_names}

    @cached_property
    def dataset_paths(self) -> Mapping[Key, Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return {k: self.DATASET_DIR / fname for k, fname in self.dataset_files.items()}

    def dataset_files_exist(self, key: Optional[Key] = None) -> bool:
        r"""Check if dataset files exist."""
        if isinstance(self.dataset_files, Mapping) and key is not None:
            return paths_exists(self.dataset_files[key])
        return paths_exists(self.dataset_paths)

    @abstractmethod
    def clean_table(self, key: Key) -> DATASET_OBJECT | None:
        r"""Create the cleaned table for the given key.

        If table is returned, the `self.serialize` method is used to write it to disk.
        If manually writing the table to disk, return None.
        """

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
            if validate and self.dataset_hashes is not NotImplemented:
                validate_file_hash(
                    self.dataset_paths[key], reference=self.dataset_hashes[key]
                )
            return

        if key is None:
            self.LOGGER.debug("Starting to clean dataset.")
            for key_ in self.table_names:
                self.clean(key=key_, force=force, validate=validate)
            self.LOGGER.debug("Finished cleaning dataset.")
            return

        self.LOGGER.debug("Starting to clean dataset <%s>", key)

        df = self.clean_table(key=key)
        if df is not None:
            self.LOGGER.info("Serializing dataset <%s>", key)
            self.serialize(df, self.dataset_paths[key])
        self.LOGGER.debug("Finished cleaning dataset <%s>", key)

        if validate and self.dataset_hashes is not NotImplemented:
            validate_file_hash(
                self.dataset_paths[key], reference=self.dataset_hashes[key]
            )

    def load_table(self, key: Key) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT.

        By default, `self.deserialize` is used to load the table from disk.
        Override this method if you want to customize loading the table from disk.
        """
        return self.deserialize(self.dataset_paths[key])

    @overload
    def load(
        self,
        *,
        key: None = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Mapping[Key, DATASET_OBJECT]:
        ...

    @overload
    def load(
        self,
        *,
        key: Key = ...,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> DATASET_OBJECT:
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

        # key=None: Recursively load all tables.
        if key is None:
            self.LOGGER.debug("Starting to load  dataset.")
            for key in self.table_names:
                self.load(key=key, force=force, validate=validate, **kwargs)
            self.LOGGER.debug("Finished loading  dataset.")
            return self.tables

        # Skip if already loaded.
        if key in self.tables and self.tables[key] is not None and not force:
            self.LOGGER.debug("Dataset already exists, skipping! <%s>", key)
            return self.tables[key]

        if validate and self.dataset_hashes is not NotImplemented:
            validate_file_hash(
                self.dataset_paths[key], reference=self.dataset_hashes[key]
            )

        # Load the table.
        self.LOGGER.debug("Starting to load  dataset <%s>", key)
        table = self.tables[key]  # implicitly calls self.load_table via LazyDict
        self.LOGGER.debug("Finished loading  dataset <%s>", key)

        if validate and self.table_hashes is not NotImplemented:
            validate_table_hash(table, reference=self.table_hashes[key])

        return table
