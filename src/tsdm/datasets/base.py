r"""Base Classes for dataset."""

__all__ = [
    # Classes
    "BaseDataset",
    "BaseDatasetMetaClass",
    "SingleFrameDataset",
    "MultiFrameDataset",
    "TimeSeriesDataset",
    "TimeSeriesCollection",
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
from collections.abc import Hashable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property, partial
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Generic, Optional, TypeAlias, overload
from urllib.parse import urlparse

import pandas
from pandas import DataFrame, Index, MultiIndex, Series
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import Self

from tsdm.config import CONFIG
from tsdm.types.aliases import Nested, PathLike
from tsdm.types.variables import KeyVar as K
from tsdm.utils import flatten_nested, paths_exists, prepend_path
from tsdm.utils.hash import hash_pandas
from tsdm.utils.remote import download
from tsdm.utils.strings import repr_dataclass, repr_mapping

DATASET_OBJECT: TypeAlias = Series | DataFrame
r"""Type hint for pandas objects."""

__logger__ = logging.getLogger(__name__)


class BaseDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # signature: type.__init__(name, bases, attributes)
        if len(args) == 1:
            attributes = {}
        elif len(args) == 3:
            _, _, attributes = args
        else:
            raise ValueError("BaseDatasetMetaClass must be used with 1 or 3 arguments.")

        if "LOGGER" not in attributes:
            cls.LOGGER = __logger__.getChild(cls.__name__)

        if "RAWDATA_DIR" not in attributes:
            if os.environ.get("GENERATING_DOCS", False):
                cls.RAWDATA_DIR = Path("~/.tsdm/rawdata/") / cls.__name__
            else:
                cls.RAWDATA_DIR = CONFIG.RAWDATADIR / cls.__name__

        if "DATASET_DIR" not in attributes:
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


class BaseDataset(ABC, metaclass=BaseDatasetMetaClass):
    r"""Abstract base class that all dataset must subclass.

    Implements methods that are available for all dataset classes.
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

    def __init__(self, *, initialize: bool = True, reset: bool = False) -> None:
        r"""Initialize the dataset."""
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
    def dataset_files(self) -> Nested[Optional[PathLike]]:
        r"""Relative paths to the cleaned dataset file(s)."""

    @property
    @abstractmethod
    def rawdata_files(self) -> Nested[Optional[PathLike]]:
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


class FrameDataset(BaseDataset, ABC):
    r"""Base class for datasets that are stored as pandas.DataFrame."""

    DEFAULT_FILE_FORMAT: str = "parquet"
    r"""Default format for the dataset."""
    RAWDATA_HASH: Optional[str | Mapping[str, str]] = None
    r"""Hash value of the raw data file(s), checked after download."""
    # DATASET_HASH: Optional[str | Mapping[str, str]] = None
    # r"""Hash value of the dataset file(s), checked after clean."""
    # TABLE_HASH: Optional[str | Mapping[str, str]] = None
    # r"""Hash value of the dataset table(s), checked after load."""

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

    def validate_table(
        self,
        table: DataFrame,
        reference: Optional[Hashable | Mapping[str, Hashable]] = None,
    ) -> None:
        r"""Validate the table."""
        filehash = hash_pandas(table)

        if reference is None:
            warnings.warn(
                f"Table {table.name!r} cannot be validated as no hash is stored in {self.__class__}."
                f"The hash is {filehash!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif isinstance(reference, int | str):
            if filehash != reference:
                warnings.warn(
                    f"Table {table.name!r} failed to validate!"
                    f"Table hash {filehash!r} does not match reference {reference!r}."
                    f"Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.LOGGER.info(
                f"Table {table.name!r} validated successfully {filehash=!r}."
            )
        elif isinstance(reference, Mapping):
            if table.name not in reference:
                warnings.warn(
                    f"Table {table.name!r} cannot be validated as it is not contained in {reference}."
                    f"The hash is {filehash!r}."
                    f"Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif table.name in reference and filehash != reference[table.name]:
                warnings.warn(
                    f"Table {table.name!r} failed to validate!"
                    f"Table hash {filehash!r} does not match reference {reference[table.name]!r}."
                    f"Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                self.LOGGER.info(
                    f"Table {table.name!r} validated successfully {filehash=!r}."
                )
        else:
            raise TypeError(f"Unsupported type for {reference=}.")

        self.LOGGER.debug("Validated %s", table.name)

    def validate(
        self,
        filespec: Nested[None | str | Path],
        /,
        *,
        reference: Optional[Hashable | Mapping[str, Hashable]] = None,
    ) -> None:
        r"""Validate the file hash."""
        self.LOGGER.debug("Validating %s", filespec)

        if isinstance(filespec, DataFrame):
            self.validate_table(filespec, reference=reference)
            return

        if isinstance(filespec, Mapping):
            for value in filespec.values():
                self.validate(value, reference=reference)
            return
        if isinstance(filespec, Sequence) and not isinstance(filespec, str | Path):
            for value in filespec:
                self.validate(value, reference=reference)
            return
        if filespec is None:
            return

        assert isinstance(filespec, str | Path), f"{filespec=} wrong type!"
        filespec = Path(filespec)
        file = Path(filespec)

        if not file.exists():
            raise FileNotFoundError(f"File {file.name!r} does not exist!")

        filehash = sha256(file.read_bytes()).hexdigest()

        if reference is None:
            warnings.warn(
                f"File {file.name!r} cannot be validated as no hash is stored in {self.__class__}."
                f"The filehash is {filehash!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif isinstance(reference, str | Path):
            if filehash != reference:
                warnings.warn(
                    f"File {file.name!r} failed to validate!"
                    f"File hash {filehash!r} does not match reference {reference!r}."
                    f"Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.LOGGER.info(
                f"File {file.name!r} validated successfully {filehash=!r}."
            )
        elif isinstance(reference, Mapping):
            if not (file.name in reference) ^ (file.stem in reference):
                warnings.warn(
                    f"File {file.name!r} cannot be validated as it is not contained in {reference}."
                    f"The filehash is {filehash!r}."
                    f"Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif file.name in reference and filehash != reference[file.name]:
                warnings.warn(
                    f"File {file.name!r} failed to validate!"
                    f"File hash {filehash!r} does not match reference {reference[file.name]!r}."
                    f"Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif file.stem in reference and filehash != reference[file.stem]:
                warnings.warn(
                    f"File {file.name!r} failed to validate!"
                    f"File hash {filehash!r} does not match reference {reference[file.stem]!r}."
                    f"Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                self.LOGGER.info(
                    f"File {file.name!r} validated successfully {filehash=!r}."
                )
        else:
            raise TypeError(f"Unsupported type for {reference=}.")

        self.LOGGER.debug("Validated %s", filespec)


class SingleFrameDataset(FrameDataset):
    r"""Dataset class that consists of a singular DataFrame."""

    __dataset: DATASET_OBJECT = NotImplemented
    DATASET_HASH: Optional[str] = None
    r"""Hash value of the dataset file(s), checked after clean."""
    TABLE_HASH: Optional[int] = None
    r"""Hash value of the table file(s), checked after load."""

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

    def download_table(self) -> None:
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

        if validate:
            self.validate(table, reference=self.TABLE_HASH)

        return table

    def clean(self, *, force: bool = True, validate: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        if self.dataset_files_exist() and not force:
            self.LOGGER.debug("Dataset files already exist, skipping.")
            return

        if not self.rawdata_files_exist():
            self.download(force=force, validate=validate)

        if validate:
            self.validate(self.rawdata_paths, reference=self.RAWDATA_HASH)

        self.LOGGER.debug("Starting to clean dataset.")
        df = self.clean_table()
        if df is not None:
            self.LOGGER.info("Serializing dataset.")
            self.serialize(df, self.dataset_paths)
        self.LOGGER.debug("Finished cleaning dataset.")

        if validate:
            self.validate(self.dataset_paths, reference=self.DATASET_HASH)

    def download(self, *, force: bool = True, validate: bool = True) -> None:
        r"""Download the dataset."""
        if self.rawdata_files_exist() and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        self.LOGGER.debug("Starting to download dataset.")
        self.download_table()
        self.LOGGER.debug("Starting downloading dataset.")

        if validate:
            self.validate(self.rawdata_paths, reference=self.RAWDATA_HASH)


class MultiFrameDataset(FrameDataset, Generic[K]):
    r"""Dataset class that consists of a multiple DataFrames.

    The Datasets are accessed by their index.
    We subclass `Mapping` to provide the mapping interface.
    """

    DATASET_HASH: Optional[Mapping[K, str]] = None
    r"""Hash value of the dataset file(s), checked after clean."""
    TABLE_HASH: Optional[Mapping[K, int]] = None
    r"""Hash value of the dataset file(s), checked after load."""

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
    def KEYS(self) -> Sequence[K]:
        r"""Return the index of the dataset."""
        # TODO: use abstract-attribute!
        # https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

    @cached_property
    def dataset(self) -> MutableMapping[K, DATASET_OBJECT]:
        r"""Store cached version of dataset."""
        return {key: None for key in self.KEYS}

    @cached_property
    def dataset_files(self) -> Mapping[K, str]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.{self.DEFAULT_FILE_FORMAT}" for key in self.KEYS}

    @cached_property
    def dataset_paths(self) -> Mapping[K, Path]:
        r"""Absolute paths to the dataset files for each key."""
        return {
            key: self.DATASET_DIR / file for key, file in self.dataset_files.items()
        }

    def rawdata_files_exist(self, key: Optional[K] = None) -> bool:
        r"""Check if raw data files exist."""
        if key is None:
            return paths_exists(self.rawdata_paths)
        if isinstance(self.rawdata_paths, Mapping):
            return paths_exists(self.rawdata_paths[key])
        return paths_exists(self.rawdata_paths)

    def dataset_files_exist(self, key: Optional[K] = None) -> bool:
        r"""Check if dataset files exist."""
        if key is None:
            return paths_exists(self.dataset_paths)
        return paths_exists(self.dataset_paths[key])

    @abstractmethod
    def clean_table(self, key: K) -> DATASET_OBJECT | None:
        r"""Clean the selected DATASET_OBJECT."""

    def clean(
        self,
        key: Optional[K] = None,
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
                self.validate(self.dataset_paths[key], reference=self.DATASET_HASH)
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
            self.validate(self.dataset_paths[key], reference=self.DATASET_HASH)

    def load_table(self, key: K) -> DATASET_OBJECT:
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
    ) -> Mapping[K, Any]:
        ...

    @overload
    def load(
        self,
        *,
        key: Optional[K] = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        ...

    def load(
        self,
        *,
        key: Optional[K] = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Mapping[K, DATASET_OBJECT] | DATASET_OBJECT:
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
            self.validate(table, reference=self.TABLE_HASH)

        return self.dataset[key]

    def download_table(self, *, key: K = NotImplemented) -> None:
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
        key: Optional[K] = None,
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
                self.validate(self.rawdata_paths, reference=self.RAWDATA_HASH)
            return

        # Download specific key
        self.LOGGER.debug("Starting to download dataset <%s>", key)
        self.download_table(key=key)
        self.LOGGER.debug("Finished downloading dataset <%s>", key)


@dataclass
class TimeSeriesDataset(TorchDataset[Series]):
    r"""Abstract Base Class for TimeSeriesDatasets.

    A TimeSeriesDataset is a dataset that contains time series data and MetaData.
    More specifically, it is a tuple (TS, M) where TS is a time series and M is metdata.
    """

    timeseries: DataFrame
    r"""The time series data."""

    _: KW_ONLY

    # Main Attributes
    name: str = NotImplemented
    r"""The name of the dataset."""
    index: Index = NotImplemented
    r"""The time-index of the dataset."""
    metadata: Optional[DataFrame] = None
    r"""The metadata of the dataset."""

    # Space Descriptors
    time_features: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    value_features: Optional[DataFrame] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    metadata_features: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.name is NotImplemented:
            self.name = self.__class__.__name__
        if self.index is NotImplemented:
            self.index = self.timeseries.index.copy().unqiue()

    def __len__(self) -> int:
        r"""Return the number of timestamps."""
        return len(self.index)

    @overload
    def __getitem__(self, key: Hashable) -> Series:
        ...

    @overload
    def __getitem__(self, key: Index | slice | list[Hashable]) -> DataFrame:
        ...

    def __getitem__(self, key):
        r"""Get item from timeseries."""
        return self.timeseries.loc[key]

    def __iter__(self) -> Iterator[Series]:
        r"""Iterate over the timestamps."""
        return iter(self.index)

    def __repr__(self) -> str:
        r"""Get the representation of the collection."""
        return repr_dataclass(self, title=self.name)


@dataclass
class TimeSeriesCollection(Mapping[Any, TimeSeriesDataset]):
    r"""Abstract Base Class for **equimodal** TimeSeriesCollections.

    A TimeSeriesCollection is a tuple (I, D, G) consiting of

    - index $I$
    - indexed TimeSeriesDatasets $D = { (TS_i, M_i) ∣ i ∈ I }$
    - global variables $G∈𝓖$
    """

    timeseries: DataFrame
    r"""The time series data."""

    _: KW_ONLY

    # Main attributes
    name: str = NotImplemented
    r"""The name of the collection."""
    index: Index = NotImplemented
    r"""The index of the collection."""

    metadata: Optional[DataFrame] = None
    r"""The metadata of the dataset."""
    global_metadata: Optional[DataFrame] = None
    r"""The global data of the dataset."""

    # Space descriptors
    index_features: Optional[DataFrame] = None
    r"""Data associated with each index such as measurement device, unit, etc."""
    time_features: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    value_features: Optional[DataFrame] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    metadata_features: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    global_features: Optional[DataFrame] = None
    r"""Data associated with each global metadata such as measurement device, unit,  etc."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.name is NotImplemented:
            if hasattr(self.timeseries, "name") and self.timeseries.name is not None:
                self.name = str(self.timeseries.name)
            else:
                self.name = self.__class__.__name__
        if self.index is NotImplemented:
            if self.metadata is not None:
                self.index = self.metadata.index.copy().unique()
            elif isinstance(self.timeseries.index, MultiIndex):
                self.index = self.timeseries.index.copy().droplevel(-1).unique()
                # self.timeseries = self.timeseries.droplevel(0)
            else:
                self.index = self.timeseries.index.copy().unique()

    @overload
    def __getitem__(self, key: K) -> TimeSeriesDataset:
        ...

    @overload
    def __getitem__(self, key: slice) -> Self:
        ...

    def __getitem__(self, key):
        r"""Get the timeseries and metadata of the dataset at index `key`."""
        # TODO: There must be a better way to slice this
        if (
            isinstance(key, Series)
            and isinstance(key.index, Index)
            and not isinstance(key.index, MultiIndex)
        ):
            ts = self.timeseries.loc[key[key].index]
        else:
            ts = self.timeseries.loc[key]

        # make sure metadata is always DataFrame.
        if self.metadata is None:
            md = None
        elif isinstance(_md := self.metadata.loc[key], Series):
            md = self.metadata.loc[[key]]
        else:
            md = _md

        if self.time_features is None:
            tf = None
        elif self.time_features.index.equals(self.index):
            tf = self.time_features.loc[key]
        else:
            tf = self.time_features

        if self.value_features is None:
            vf = None
        elif self.value_features.index.equals(self.index):
            vf = self.value_features.loc[key]
        else:
            vf = self.value_features

        if self.metadata_features is None:
            mf = None
        elif self.metadata_features.index.equals(self.index):
            mf = self.metadata_features.loc[key]
        else:
            mf = self.metadata_features

        if isinstance(ts.index, MultiIndex):
            index = ts.index.droplevel(-1).unique()
            return TimeSeriesCollection(
                name=self.name,
                index=index,
                timeseries=ts,
                metadata=md,
                time_features=tf,
                value_features=vf,
                metadata_features=mf,
                global_metadata=self.global_metadata,
                global_features=self.global_features,
                index_features=self.index_features,
            )

        return TimeSeriesDataset(
            name=self.name,
            index=ts.index,
            timeseries=ts,
            metadata=md,
            time_features=tf,
            value_features=vf,
            metadata_features=mf,
        )

    def __len__(self) -> int:
        r"""Get the length of the collection."""
        return len(self.index)

    def __iter__(self) -> Iterator[K]:
        r"""Iterate over the collection."""
        return iter(self.index)
        # for key in self.index:
        #     yield self[key]

    def __repr__(self) -> str:
        r"""Get the representation of the collection."""
        return repr_dataclass(self, title=self.name)
