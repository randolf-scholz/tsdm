r"""Base Classes for dataset."""

# NOTE: signature of metaclass.__init__ should match that of type.__new__
# NOTE: type.__new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> Self
# NOTE: type.__init__(self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> None

__all__ = [
    # ABCs & Protocols
    "BaseDataset",
    "BaseDatasetMetaClass",
    "Dataset",
    "MultiTableDataset",
    "SingleTableDataset",
]

import html
import inspect
import logging
import os
import shutil
import subprocess
import warnings
import webbrowser
from abc import abstractmethod
from collections.abc import Collection, Iterator, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from urllib.parse import urlparse

import pandas
from pyarrow import Table, parquet
from tqdm.autonotebook import tqdm
from typing_extensions import (
    Any,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    overload,
    runtime_checkable,
)

from tsdm.config import CONFIG
from tsdm.testing.hash import validate_file_hash, validate_table_hash
from tsdm.types.aliases import PathLike
from tsdm.types.variables import T_co, str_var as Key
from tsdm.utils import paths_exists, remote
from tsdm.utils.decorators import wrap_method
from tsdm.utils.funcutils import get_return_typehint
from tsdm.utils.lazydict import LazyDict, LazyValue
from tsdm.utils.pprint import repr_array, repr_mapping


@runtime_checkable
class Dataset(Protocol[T_co]):
    """Protocol for Dataset."""

    # TODO: make a bug report. Mypy does not honor the type hint for INFO_URL
    # INFO_URL: ClassVar[Optional[str]]
    # r"""HTTP address containing additional information about the dataset."""
    SOURCE_URL: ClassVar[str] = NotImplemented
    r"""Web address from where the dataset can be downloaded."""
    INFO_URL: ClassVar[str] = NotImplemented
    r"""Web address containing documentational information about the dataset."""
    RAWDATA_DIR: ClassVar[Path]
    r"""Directory where the raw data is stored."""
    DATASET_DIR: ClassVar[Path]
    r"""Directory where the dataset is stored."""

    @classmethod
    def info(cls) -> None:
        r"""Open dataset information in browser."""
        if cls.INFO_URL is NotImplemented:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    @property
    @abstractmethod
    def rawdata_files(self) -> Sequence[str]:
        """Return list of file names that make up the raw data."""
        ...

    @property
    def rawdata_paths(self) -> Mapping[str, Path]:
        """Return mapping from filenames to paths to the rawdata files."""
        return {
            str(fname): (self.RAWDATA_DIR / fname).absolute()
            for fname in self.rawdata_files
        }

    rawdata_files: Sequence[str]  # type: ignore[no-redef]
    # CF. https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020
    rawdata_paths: Mapping[str, Path]  # type: ignore[no-redef]
    # CF. https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020

    @abstractmethod
    def download(self) -> None:
        """Download the dataset."""
        ...

    @abstractmethod
    def clean(self) -> None:
        """Clean the dataset."""
        ...

    @abstractmethod
    def load(self) -> T_co:
        """Load the dataset."""
        ...


class BaseDatasetMetaClass(type(Protocol)):  # type: ignore[misc]
    r"""Metaclass for BaseDataset."""

    def __init__(
        self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            self.LOGGER = logging.getLogger(f"{self.__module__}.{self.__name__}")

        if "RAWDATA_DIR" not in namespace:
            self.RAWDATA_DIR = (
                Path("~/.tsdm/rawdata/")
                if os.environ.get("GENERATING_DOCS", False)
                else CONFIG.RAWDATADIR
            ) / self.__name__

        if "DATASET_DIR" not in namespace:
            self.DATASET_DIR = (
                Path("~/.tsdm/datasets")
                if os.environ.get("GENERATING_DOCS", False)
                else CONFIG.DATASETDIR
            ) / self.__name__

        # add post_init hook
        self.__init__ = wrap_method(self.__init__, after=self.__post_init__)  # type: ignore[misc]


class BaseDataset(Dataset[T_co], metaclass=BaseDatasetMetaClass):
    r"""Abstract base class that all datasets must subclass.

    Implements methods that are available for all dataset classes.


    We allow for systematic validation of the datasets.
    Users can check the validity of the datasets by implementing the following attributes/properites:

    - rawdata_hashes: Used to verify rawdata files after `download` and before `clean`.
    - dataset_hashes: Used to verify cleaned dataset files after `clean` and before `load`.
    - table_hashes: Used to verify in-memory tables after `load`.
    - table_schemas: Used to verify in-memory tables after `load`.
    """

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the dataset."""
    DEFAULT_FILE_FORMAT: ClassVar[str] = "parquet"
    r"""Default format for the dataset."""

    SOURCE_URL: ClassVar[str] = NotImplemented
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL: ClassVar[str] = NotImplemented
    r"""HTTP address containing additional information about the dataset."""
    RAWDATA_DIR: ClassVar[Path]
    r"""Location where the raw data is stored."""
    DATASET_DIR: ClassVar[Path]
    r"""Location where the pre-processed data is stored."""

    # instance attribute
    __version__: str = NotImplemented
    r"""Version of the dataset, keep empty for unversioned dataset."""
    rawdata_hashes: Mapping[str, str | None] = NotImplemented
    r"""Hashes of the raw dataset file(s)."""
    rawdata_schemas: Mapping[str, Mapping[str, str]] = NotImplemented
    r"""Schemas for the raw dataset tables(s)."""
    rawdata_shapes: Mapping[str, tuple[int, ...]] = NotImplemented
    r"""Shapes for the raw dataset tables(s)."""

    def __init__(
        self,
        *,
        initialize: bool = True,
        reset: bool = False,
        verbose: bool = False,
        version: str = NotImplemented,
    ) -> None:
        r"""Initialize the dataset.

        Args:
            initialize: Whether to initialize the dataset.
            reset: Whether to reset the dataset.
            version: Version of the dataset. Leave empty for unversioned dataset.
            verbose: Whether to print verbose output.
        """
        self.verbose = verbose
        self.initialize = initialize

        # set the version
        if version is not NotImplemented:
            self.__version__ = str(version)

        # update the paths
        if self.__version__ is not NotImplemented:
            self.RAWDATA_DIR /= self.__version__  # type: ignore[misc]
            self.DATASET_DIR /= self.__version__  # type: ignore[misc]

        if not inspect.isabstract(self):
            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
            self.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        if reset:
            # delete rawdata and dataset directories
            self.remove_rawdata_files()
            self.remove_dataset_files()

    def __post_init__(self) -> None:
        r"""Initialize the dataset."""
        if self.initialize:
            # NOTE: We call clean first for memory efficiency.
            #  Preprocessing can take lots of resources, so we should only load tables
            #  on a need-to-know basis.
            self.clean()
            self.load(initializing=True)

    def __repr__(self) -> str:
        r"""Return a string representation of the dataset."""
        return f"{self.__class__.__name__}()"

    @staticmethod
    def serialize(table: T_co, path: Path, /, **kwargs: Any) -> None:  # type: ignore[misc]
        r"""Serialize the dataset."""
        # NOTE: This should be type safe even for covariant types.
        file_type = path.suffix
        assert file_type.startswith("."), "File must have a suffix!"
        file_type = file_type[1:]

        if isinstance(table, Table) and file_type == "parquet":
            parquet.write_table(table, path, **kwargs)
            return

        # check if table has a custom writer.
        if hasattr(table, f"to_{file_type}"):
            writer = getattr(table, f"to_{file_type}")
            writer(path, **kwargs)
            return

        raise NotImplementedError(f"No loader for {file_type=}")

    @staticmethod
    def deserialize(path: Path, /, **kwargs: Any) -> T_co:
        r"""Deserialize the dataset."""
        file_type = path.suffix
        assert file_type.startswith("."), "File must have a suffix!"
        file_type = file_type[1:]
        pandas_version = tuple(int(i) for i in pandas.__version__.split("."))

        if hasattr(pandas, f"read_{file_type}"):
            loader = getattr(pandas, f"read_{file_type}")

            if file_type in {"parquet", "feather"}:
                if "engine" not in kwargs:
                    kwargs["engine"] = "pyarrow"
                if "dtype_backend" not in kwargs and pandas_version >= (2, 0, 0):
                    kwargs["dtype_backend"] = "pyarrow"
                table = loader(path, **kwargs)
            else:
                table = loader(path, **kwargs)

            return table.squeeze()

        raise NotImplementedError(f"No loader for {file_type=}")

    @property
    def version_info(self) -> tuple[int, ...]:
        r"""Version information of the dataset."""
        if self.__version__ is NotImplemented:
            return NotImplemented
        return tuple(int(i) for i in self.__version__.split("."))

    @cached_property
    def rawdata_valid(self) -> bool:
        r"""Check if raw data files exist."""
        self.LOGGER.debug("Validating raw data files.")
        validate_file_hash(self.rawdata_paths, self.rawdata_hashes)
        return True

    def rawdata_files_exist(self, key: Optional[str] = None) -> bool:
        r"""Check if raw data files exist."""
        if key is None:
            return paths_exists(self.rawdata_paths)
        if key not in self.rawdata_paths:
            raise KeyError(f"{key=} not in {self.rawdata_paths=}")
        return paths_exists(self.rawdata_paths[key])

    @abstractmethod
    def clean(self) -> None:
        r"""Clean an already downloaded raw dataset and stores it in self.data_dir.

        Preferably, use the '.parquet' data format.
        """
        ...

    @abstractmethod
    def load(self, *, initializing: bool = False) -> T_co:
        r"""Load the pre-processed dataset."""
        ...

    def download_from_url(self, url: str, path: PathLike, /, **options: Any) -> None:
        r"""Download files from a URL."""
        parsed_url = urlparse(url)
        path = Path(path)
        target_directory = path.parent

        match parsed_url.netloc:
            case "www.kaggle.com":
                # change options to kaggle cli options
                opts = " ".join(f"--{k} {v}" for k, v in options.items())
                kaggle_name = Path(parsed_url.path).name
                subprocess.run(
                    "kaggle competitions download"
                    f" -p {target_directory} -c {kaggle_name} {opts}",
                    shell=True,
                    check=True,
                )
            case "github.com":
                opts = " ".join(f"--{k} {v}" for k, v in options.items())
                svn_url = url.replace("tree/main", "trunk")
                subprocess.run(
                    f"svn export --force {svn_url} {target_directory} {opts}",
                    shell=True,
                    check=True,
                )
            case _:  # default parsing, including for UCI dataset
                remote.download(url, path, **options)

    def download_file(self, fname: str, /) -> None:
        r"""Download a single rawdata file.

        Override this method for custom download logic.
        """
        if self.SOURCE_URL is NotImplemented:
            self.LOGGER.debug("Dataset provides no base_url. Assumed offline")
            return

        url = self.SOURCE_URL + fname
        path = self.RAWDATA_DIR / fname
        self.LOGGER.debug("Downloading from %s", url)
        self.download_from_url(url, path)

    def download(
        self,
        *,
        key: Optional[str] = None,
        force: bool = True,
        validate: bool = True,
    ) -> None:
        r"""Download the dataset."""
        # Recurse if key is None.
        if key is None:
            self.LOGGER.debug("Starting to download dataset.")
            for _key in self.rawdata_paths:
                self.download(key=_key, force=force, validate=validate)
            self.LOGGER.debug("Finished downloading dataset.")
            return

        # Check if the file already exists.
        if self.rawdata_files_exist(key) and not force:
            self.LOGGER.debug("Dataset already exists. Skipping download.")
            return

        # Download the file.
        self.LOGGER.debug("Starting to download dataset <%s>", key)
        self.download_file(key)
        self.LOGGER.debug("Finished downloading dataset <%s>", key)

        # Validate the file.
        if validate and self.rawdata_hashes is not NotImplemented:
            validate_file_hash(self.rawdata_paths[key], self.rawdata_hashes)

    def remove_rawdata_files(self, *, ask_permission: bool = True) -> None:
        r"""Recreate the rawdata directory."""
        if not self.RAWDATA_DIR.exists():
            raise FileNotFoundError(f"{self.RAWDATA_DIR} does not exist!")

        if (
            ask_permission
            and input(f"Delete {self.RAWDATA_DIR}? [y/N] ").lower() == "y"
        ):
            return

        try:  # remove the rawdata directory
            shutil.rmtree(self.RAWDATA_DIR)
        except Exception as exc:
            raise RuntimeError(f"Failed to delete {self.RAWDATA_DIR}") from exc

        # recreate the rawdata directory
        self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)

    def remove_dataset_files(self, *, ask_permission: bool = True) -> None:
        r"""Recreate the dataset directory."""
        if not self.DATASET_DIR.exists():
            raise FileNotFoundError(f"{self.DATASET_DIR} does not exist!")

        if (
            ask_permission
            and input(f"Delete {self.DATASET_DIR}? [y/N] ").lower() == "y"
        ):
            return

        try:  # remove the dataset directory
            shutil.rmtree(self.DATASET_DIR)
        except Exception as exc:
            raise RuntimeError(f"Failed to delete {self.DATASET_DIR}") from exc

        # recreate the dataset directory
        self.DATASET_DIR.mkdir(parents=True, exist_ok=True)


class SingleTableDataset(BaseDataset[T_co]):
    r"""Dataset class that consists of a singular DataFrame."""

    RAWDATA_DIR: ClassVar[Path]
    """Path to raw data directory."""
    DATASET_DIR: ClassVar[Path]
    """Path to pre-processed data directory."""

    _table: T_co = NotImplemented
    """INTERNAL: the dataset."""

    # Validation - Implement on per dataset basis!
    dataset_hash: str = NotImplemented
    r"""Hashes of the cleaned dataset file(s)."""
    table_hash: str = NotImplemented
    r"""Hashes of the in-memory cleaned dataset table(s)."""
    table_schema: Mapping[str, str] = NotImplemented
    r"""Schema of the in-memory cleaned dataset table(s)."""
    table_shape: tuple[int, ...] = NotImplemented
    r"""Shape of the in-memory cleaned dataset table(s)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(table: {repr_array(self.table)})"

    def _repr_html_(self) -> str:
        if self._table is NotImplemented:
            return f"<pre>{html.escape(repr(self))}</pre>"
        if hasattr(self.table, "_repr_html_"):
            header = f"<pre>{html.escape(repr(self))}</pre>"
            # noinspection PyProtectedMember
            html_repr = self.table._repr_html_()  # pylint: disable=protected-access
            return header + html_repr
        raise NotImplementedError

    @cached_property
    def table(self) -> T_co:
        r"""Store cached version of dataset."""
        if self._table is NotImplemented:
            self._table = self.load(initializing=True)
        return self._table

    @cached_property
    def dataset_file(self) -> PathLike:
        r"""Return the dataset files."""
        return f"{self.__class__.__name__}.{self.DEFAULT_FILE_FORMAT}"

    @cached_property
    def dataset_path(self) -> Path:
        r"""Path to raw data."""
        return self.DATASET_DIR / self.dataset_file

    def dataset_file_exists(self) -> bool:
        r"""Check if the dataset file exists."""
        return paths_exists(self.dataset_path)

    @abstractmethod
    def clean_table(self) -> T_co | None:
        r"""Generate the cleaned dataset table.

        If a table is returned, the `self.serialize` method is used to write it to disk.
        If manually writing the table to disk, return None.
        """

    def load_table(self) -> T_co:
        r"""Load the dataset.

        By default, `self.deserialize` is used to load the table from disk.
        Override this method if you want to customize loading the table from disk.
        """
        return self.deserialize(self.dataset_path)

    def load(
        self,
        *,
        initializing: bool = False,
        force: bool = True,
        validate: bool = True,
    ) -> T_co:
        r"""Load the selected DATASET_OBJECT."""
        # Create the pre-processed dataset file if it doesn't exist.
        if not self.dataset_file_exists():
            self.clean(force=force, validate=validate)

        # Skip if already loaded.
        if not initializing:
            return self.table

        # Validate file if hash is provided.
        if validate and self.dataset_hash is not NotImplemented:
            validate_file_hash(self.dataset_path, self.dataset_hash)

        # Load table.
        self.LOGGER.debug("Starting to load dataset.")
        table = self.load_table()
        self.LOGGER.debug("Finished loading dataset.")

        # Validate table if hash/schema is provided.
        if validate and self.table_hash is not NotImplemented:
            validate_table_hash(table, self.table_hash)

        return table

    def clean(self, *, force: bool = False, validate: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        # Skip if dataset file already exists.
        if self.dataset_file_exists() and not force:
            self.LOGGER.debug("Dataset files already exist, skipping.")
            return

        # Download raw data if it does not exist.
        if not self.rawdata_files_exist():
            self.download(force=force, validate=validate)

        # Validate raw data.
        if validate and self.rawdata_hashes is not NotImplemented:
            assert self.rawdata_valid

        # Clean dataset.
        self.LOGGER.debug("Starting to clean dataset.")
        df = self.clean_table()
        if df is not None:
            self.LOGGER.debug("Serializing dataset.")
            self.serialize(df, self.dataset_path)
        self.LOGGER.debug("Finished cleaning dataset.")

        # Validate pre-processed file.
        if validate and self.dataset_hash is not NotImplemented:
            validate_file_hash(self.dataset_path, self.dataset_hash)


class MultiTableDataset(
    BaseDataset[dict[Key, T_co]], Mapping[Key, T_co], Generic[Key, T_co]
):
    r"""Dataset class that consists of multiple tables.

    The tables are stored in a dictionary-like object.
    """

    RAWDATA_DIR: ClassVar[Path]
    """Path to raw data directory."""
    DATASET_DIR: ClassVar[Path]
    """Path to pre-processed data directory."""

    # Validation - Implement on per dataset basis!
    dataset_hashes: Mapping[Key, str | None] = NotImplemented
    r"""Hashes of the cleaned dataset file(s)."""
    table_hashes: Mapping[Key, str | None] = NotImplemented
    r"""Hashes of the in-memory cleaned dataset table(s)."""
    table_schemas: Mapping[Key, Mapping[str, str]] = NotImplemented
    r"""Schemas of the in-memory cleaned dataset table(s)."""
    table_shapes: Mapping[Key, tuple[int, ...]] = NotImplemented
    r"""Shapes of the in-memory cleaned dataset table(s)."""

    @cached_property
    def _key_attributes(self) -> bool:
        r"""Whether to add table names as attributes."""
        if invalid_keys := {key for key in self.table_names if not key.isidentifier()}:
            warnings.warn(
                "Not adding keys as attributes!"
                f" Keys {invalid_keys} are not valid identifiers!",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        def attr_exists(obj: object, key: str) -> bool:
            """Test if attribute exists using only __getattribute__ and not __getattr__."""
            try:
                obj.__getattribute__(key)
            except AttributeError:
                return False
            return True

        if hasattr_keys := {key for key in self.table_names if attr_exists(self, key)}:
            warnings.warn(
                "Not adding keys as attributes!"
                f" Keys {hasattr_keys} already exist as attributes!",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        self.LOGGER.debug("Adding keys as attributes.")
        return True

    def __dir__(self) -> list[str]:
        if self._key_attributes:
            return list(super().__dir__()) + list(self.table_names)
        return list(super().__dir__())

    def __getattr__(self, key):
        r"""Get attribute."""
        if self._key_attributes and key in self.table_names:
            return self.tables[key]
        return self.__getattribute__(key)

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.tables)

    def __getitem__(self, key: Key, /) -> T_co:
        r"""Return the sample at index `idx`."""
        # need to manually raise KeyError otherwise __getitem__ will execute.
        if key not in self.tables:
            raise KeyError(f"Key {key} not in {self.__class__.__name__}!")
        return self.tables[key]

    def __iter__(self) -> Iterator[Key]:
        r"""Return an iterator over the dataset."""
        return iter(self.tables)

    def __repr__(self) -> str:
        r"""Pretty Print."""
        return repr_mapping(self.tables, wrapped=self)

    @property
    @abstractmethod
    def table_names(self) -> Collection[Key]:  # pyright: ignore
        r"""Return the index of the dataset."""
        # TODO: use abstract-attribute!
        # https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

    table_names: Collection[Key]  # type: ignore[no-redef]
    # Cf. https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020

    @cached_property
    def tables(self) -> dict[Key, T_co]:
        r"""Store cached version of dataset."""
        # (self.load, (key,), {}) → self.load(key=key) when tables[key] is accessed.

        if isinstance(self.table_names, Mapping):
            type_hints = self.table_names
        else:
            return_type = get_return_typehint(self.clean_table)
            type_hints = {key: str(return_type) for key in self.table_names}

        return LazyDict({
            key: LazyValue(
                self.load,
                args=(key,),
                kwargs={"initializing": True},
                type_hint=type_hints[key],
            )
            for key in self.table_names
        })

    @cached_property
    def dataset_files(self) -> dict[Key, str]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.{self.DEFAULT_FILE_FORMAT}" for key in self.table_names}

    @cached_property
    def dataset_paths(self) -> dict[Key, Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return {k: self.DATASET_DIR / fname for k, fname in self.dataset_files.items()}

    def dataset_files_exist(self, key: Optional[Key] = None) -> bool:
        r"""Check if dataset files exist."""
        if isinstance(self.dataset_paths, Mapping) and key is not None:
            return paths_exists(self.dataset_paths[key])
        return paths_exists(self.dataset_paths)

    @abstractmethod
    def clean_table(self, key: Key) -> T_co | None:
        r"""Create the cleaned table for the given key.

        If a table is returned, the `self.serialize` method is used to write it to disk.
        If manually writing the table to disk, return None.
        """
        ...

    def clean(
        self,
        key: Optional[Key] = None,
        *,
        force: bool = False,
        validate: bool = True,
        validate_rawdata: bool = True,
    ) -> None:
        r"""Create the preprocessed table for the selected key.

        Args:
            key: The key of the dataset to clean. If None, clean all datasets.
            force: Force cleaning of dataset.
            validate: Validate the dataset after cleaning.
            validate_rawdata: Validate the raw data files before cleaning.
        """
        # download raw data files if they don't exist
        if validate_rawdata and not self.rawdata_files_exist():
            self.LOGGER.debug("Raw files missing, fetching them now!")
            self.download(force=force, validate=validate)

        # validate the raw data files
        if validate_rawdata and self.rawdata_hashes is not NotImplemented:
            assert self.rawdata_valid

        # skip if cleaned files already exist
        if not force and self.dataset_files_exist(key=key):
            self.LOGGER.debug("Cleaned data file already exists, skipping <%s>", key)
            return

        # key=None: Recursively clean all tables
        if key is None:
            self.LOGGER.debug("Starting to clean dataset.")
            for name in (
                pbar := tqdm(
                    self.table_names,
                    desc="Cleaning tables",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_postfix(table=name)
                self.clean(
                    key=name, force=force, validate=validate, validate_rawdata=False
                )
            self.LOGGER.debug("Finished cleaning dataset.")
            return

        # Clean the selected table
        self.LOGGER.debug("Starting to clean dataset <%s>", key)
        df = self.clean_table(key=key)
        if df is not None:
            self.LOGGER.debug("Serializing dataset <%s>", key)
            self.serialize(df, self.dataset_paths[key])
        self.LOGGER.debug("Finished cleaning dataset <%s>", key)

        # Validate the cleaned table
        if validate and self.dataset_hashes is not NotImplemented:
            validate_file_hash(self.dataset_paths[key], self.dataset_hashes[key])

    def load_table(self, key: Key) -> T_co:
        r"""Load the selected DATASET_OBJECT.

        By default, `self.deserialize` is used to load the table from disk.
        Override this method if you want to customize loading the table from disk.
        """
        return self.deserialize(self.dataset_paths[key])

    @overload
    def load(
        self,
        *,
        force: bool = ...,
        validate: bool = ...,
        initializing: bool = ...,
    ) -> dict[Key, T_co]: ...
    @overload
    def load(
        self,
        key: Key,
        *,
        force: bool = ...,
        validate: bool = ...,
        initializing: bool = ...,
    ) -> T_co: ...
    def load(
        self,
        key: Optional[Key] = None,
        *,
        force: bool = False,
        validate: bool = True,
        initializing: bool = False,
    ) -> T_co | dict[Key, T_co]:
        r"""Load the selected DATASET_OBJECT.

        Args:
            key: The key associated with the datset
            force: Reload the dataset even if it already exists.
            validate: Validate the dataset against hash.
            initializing: Whether this is the first time the dataset is being loaded.

        Returns:
            The loaded dataset
        """
        # key=None: Recursively load all tables.
        if key is None:
            self.LOGGER.debug("Starting to load dataset.")
            for name in (
                pbar := tqdm(
                    self.table_names,
                    desc="Loading tables",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_postfix(table=name)
                self.load(key=name, force=force, validate=validate)
            self.LOGGER.debug("Finished loading dataset.")
            return self.tables

        # Skip if already loaded.
        if not force and not initializing:
            return self.tables[key]

        # Create the pre-processed dataset file if it doesn't exist.
        if not self.dataset_files_exist(key=key):
            self.clean(key=key, force=force, validate=validate)

        # Validate file if hash is provided.
        if validate and self.dataset_hashes is not NotImplemented:
            validate_file_hash(self.dataset_paths[key], self.dataset_hashes[key])

        # Load the table, make sure to use the cached version if it exists.
        self.LOGGER.debug("Starting to load dataset <%s>", key)
        table = self.load_table(key)
        self.LOGGER.debug("Finished loading dataset <%s>", key)

        # Validate the loaded table.
        if validate and self.table_hashes is not NotImplemented:
            validate_table_hash(table, self.table_hashes[key])

        return table
