r"""Base Classes for dataset."""

# NOTE: signature of metaclass.__init__ should match that of type.__new__
# NOTE: type.__new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> Self
# NOTE: type.__init__(self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> None

__all__ = [
    # ABCs & Protocols
    "Dataset",
    "DatasetBase",
]

import inspect
import logging
import os
import shutil
import warnings
import webbrowser
from abc import abstractmethod
from collections.abc import Collection, Iterator, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Optional,
    Protocol,
    Self,
    _ProtocolMeta as ProtocolMeta,
    cast,
    final,
    overload,
    runtime_checkable,
)
from zipfile import ZipFile

from tqdm.auto import tqdm

from tsdm.config import CONFIG
from tsdm.constants import EMPTY_MAP, NOT_GIVEN
from tsdm.data import serialize
from tsdm.testing.hashutils import (
    ErrorHandler,
    ValidationError,
    validate_file,
    validate_table,
)
from tsdm.types.aliases import FilePath
from tsdm.utils import paths_exists, remote
from tsdm.utils.contextmanagers import timer
from tsdm.utils.funcutils import get_return_typehint
from tsdm.utils.lazydict import LazyDict
from tsdm.utils.pprint import repr_mapping
from tsdm.utils.system import query_bool


@runtime_checkable
class Dataset[Key, T](Protocol):  # +T
    r"""Protocol for Dataset.

    A dataset is a collection of table-like objects indexed by keys.

    This protocol describes a reduced interface of the `DatasetBase` class,
    and covers only methods that should be used at call sites, that is methods
    from an already instantiated object.
    """

    SOURCE_URL: ClassVar[str] = NOT_GIVEN
    r"""Web address from where the dataset can be downloaded."""
    INFO_URL: ClassVar[Optional[str]] = None
    r"""Web address containing documentational information about the dataset."""
    RAWDATA_DIR: ClassVar[Path]
    r"""Directory where the raw data is stored."""
    DATASET_DIR: ClassVar[Path]
    r"""Directory where the dataset is stored."""

    # region property/attributes -------------------------------------------------------
    @property
    def __version__(self) -> str | None: ...  # pyright: ignore[reportRedeclaration]

    # SEE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020
    __version__: str | None  # type: ignore[no-redef]

    @property
    @abstractmethod
    def rawdata_files(self) -> Sequence[str]:  # pyright: ignore[reportRedeclaration]
        r"""Return list of file names that make up the raw data."""
        ...

    # SEE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020
    rawdata_files: Sequence[str] | cached_property[Sequence[str]]  # type: ignore[no-redef]

    @property
    @abstractmethod
    def table_names(self) -> Collection[Key]:  # pyright: ignore[reportRedeclaration]
        r"""READ-ONLY: The names of the tables."""

    # SEE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020
    table_names: Collection[Key] | cached_property[Collection[Key]]  # type: ignore[no-redef]

    @property
    @abstractmethod
    def tables(self) -> Mapping[Key, T]: ...  # pyright: ignore[reportRedeclaration]

    # SEE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020
    tables: Mapping[Key, T] | cached_property[Mapping[Key, T]]  # type: ignore[no-redef]
    # endregion property/attributes ----------------------------------------------------

    @property
    def rawdata_paths(self) -> Mapping[str, Path]:
        r"""Return mapping from filenames to paths to the rawdata files."""
        return {
            str(fname): (self.RAWDATA_DIR / fname).absolute()
            for fname in self.rawdata_files
        }

    @classmethod
    def info(cls) -> None:
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            raise NotImplementedError("No INFO_URL provided for this dataset!")
        webbrowser.open_new_tab(cls.INFO_URL)

    @classmethod
    @abstractmethod
    def deserialize(cls, filepath: FilePath, /) -> Self:
        r"""Deserialize the dataset (from cleaned format)."""
        ...

    @abstractmethod
    def serialize(self, filepath: FilePath, /) -> None:
        r"""Serialize the (cleaned) dataset to a specific path."""
        ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Key]: ...
    def __getitem__(self, key: Key, /) -> T: ...
    def __contains__(self, key: object, /) -> bool: ...


class _DatasetMeta(ProtocolMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(
        cls,  # noqa: N805
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> None:
        r"""When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if "RAWDATA_DIR" not in namespace:
            cls.RAWDATA_DIR = (
                Path("~/.tsdm/rawdata/")
                if os.environ.get("GENERATING_DOCS", False)
                else CONFIG.RAWDATADIR
            ) / cls.__name__

        if "DATASET_DIR" not in namespace:
            cls.DATASET_DIR = (
                Path("~/.tsdm/datasets")
                if os.environ.get("GENERATING_DOCS", False)
                else CONFIG.DATASETDIR
            ) / cls.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""When an instance of the class is created, this method is called."""
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class DatasetBase[Key: str, T](
    Mapping[Key, T], Dataset[Key, T], metaclass=_DatasetMeta
):  # Key, +T
    r"""Abstract base class that all datasets must subclass.

    Implements methods that are available for all dataset classes.

    We allow for systematic validation of the datasets.
    Users can check the validity of the datasets by implementing the following attributes/properites:

    - rawdata_hashes: Used to verify rawdata files after `download` and before `clean`.
    - dataset_hashes: Used to verify cleaned dataset files after `clean` and before `load`.
    - table_hashes: Used to verify in-memory tables after `load`.
    - table_schemas: Used to verify in-memory tables after `load`.
    """

    # region class attributes ----------------------------------------------------------
    # Abstract members: table_names, clean_table
    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the dataset."""
    DEFAULT_FILE_FORMAT: ClassVar[str] = "parquet"
    r"""Default format for the dataset."""
    SOURCE_URL: ClassVar[str] = NOT_GIVEN
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL: ClassVar[Optional[str]] = None
    r"""HTTP address containing additional information about the dataset."""
    RAWDATA_DIR: ClassVar[Path]
    r"""Location where the raw data is stored."""
    DATASET_DIR: ClassVar[Path]
    r"""Location where the pre-processed data is stored."""
    # endregion class attributes -------------------------------------------------------

    # region instance attributes -------------------------------------------------------
    # FIXME: Use Read-Only type
    __version__: str | None = None
    r"""READ-ONLY: The version of the dataset."""
    rawdata_hashes: Mapping[str, str | None] = EMPTY_MAP
    r"""Hashes of the raw dataset file(s)."""
    rawdata_schemas: Mapping[str, Mapping[str, str]] = EMPTY_MAP
    r"""Schemas for the raw dataset tables(s)."""
    rawdata_shapes: Mapping[str, tuple[int, ...]] = EMPTY_MAP
    r"""Shapes for the raw dataset tables(s)."""
    dataset_hashes: Mapping[Key, str | None] = EMPTY_MAP
    r"""Hashes of the cleaned dataset file(s)."""
    table_hashes: Mapping[Key, str | None] = EMPTY_MAP
    r"""Hashes of the in-memory cleaned dataset table(s)."""
    table_schemas: Mapping[Key, Mapping[str, str]] = EMPTY_MAP
    r"""Schemas of the in-memory cleaned dataset table(s)."""
    table_shapes: Mapping[Key, tuple[int, ...]] = EMPTY_MAP
    r"""Shapes of the in-memory cleaned dataset table(s)."""
    # endregion instance attributes ----------------------------------------------------

    tables: dict[Key, T]
    r"""INTERNAL: the dataset."""

    # region constructors --------------------------------------------------------------
    @classmethod
    def from_tables(cls, tables: Mapping[Key, T], /) -> Self:
        r"""Create a dataset from a table."""
        obj = cls(initialize=False)
        obj.tables = dict(tables)
        return obj

    def __init__(
        self,
        *,
        initialize: bool = True,
        reset: bool = False,
        verbose: bool = True,
        version: Optional[str] = None,
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
        if self.__version__ is None and version is not None:
            self.__version__ = str(version)

        # update the paths
        if self.__version__ is not None:
            self.RAWDATA_DIR /= self.__version__  # type: ignore[misc]
            self.DATASET_DIR /= self.__version__  # type: ignore[misc]

        if not inspect.isabstract(self):
            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
            self.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        if reset:
            # delete rawdata and dataset directories
            self.reset_rawdata_files()
            self.reset_dataset_files()

        # initialize tables
        self.tables = LazyDict.from_func(
            self.table_names,
            self.load,
            kwargs={"initializing": True},
            type_hint=get_return_typehint(self.clean_table),
        )

    def __post_init__(self) -> None:
        r"""Initialize the dataset."""
        if self.initialize:
            # NOTE: We call clean first for memory efficiency.
            #  Preprocessing can take lots of resources, so we should only load tables
            #  on a need-to-know basis.
            self.clean()
            self.load(initializing=True)

    # endregion constructors -----------------------------------------------------------

    # region serialization methods -----------------------------------------------------
    serialize_table = staticmethod(serialize.serialize_table)
    deserialize_table = staticmethod(serialize.deserialize_table)

    @classmethod
    def deserialize(cls, filepath: FilePath, /) -> Self:
        r"""Deserialize the dataset."""
        tables: dict[Key, T] = {}
        with ZipFile(filepath) as archive:
            for fname in archive.namelist():
                with archive.open(fname) as file:
                    name = cast(Key, Path(fname).stem)
                    extension = Path(fname).suffix[1:]
                    tables[name] = cls.deserialize_table(file, loader=extension)

        return cls.from_tables(tables)

    def serialize(self, filepath: FilePath, /) -> None:
        r"""Serialize the dataset."""
        path = Path(filepath)
        if path.suffix != ".zip":
            raise ValueError("Path must be a zip file if serializing whole dataset.")
        with ZipFile(path, "w") as archive:
            extension = self.DEFAULT_FILE_FORMAT
            for name, table in self.items():
                fname = f"{name}.{extension}"
                with archive.open(fname, "w") as file:
                    self.serialize_table(table, file, writer=extension)

    # endregion serialization methods --------------------------------------------------

    # region reset methods -------------------------------------------------------------
    @classmethod
    def reset(cls, *, force: bool = False) -> None:
        r"""Reset the dataset."""
        try:
            self = cls(initialize=False)
        except Exception as exc:
            exc.add_note("Could not create instance of dataset!")
            raise

        self.reset_rawdata_files(force=force)
        self.reset_dataset_files(force=force)

    def reset_rawdata_files(self, *, force: bool = False) -> None:
        r"""Recreate the rawdata directory."""
        if not self.RAWDATA_DIR.exists():
            raise FileNotFoundError(f"{self.RAWDATA_DIR} does not exist!")

        if force or query_bool(f"Delete {self.RAWDATA_DIR}?", default=False):
            try:  # remove the rawdata directory
                shutil.rmtree(self.RAWDATA_DIR)
            except Exception as exc:
                raise RuntimeError(f"Failed to delete {self.RAWDATA_DIR}") from exc

            # recreate the rawdata directory
            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
            return

        # else do nothing
        self.LOGGER.debug("Rawdata files not deleted.")

    def reset_dataset_files(self, *, force: bool = False) -> None:
        r"""Recreate the dataset directory."""
        if not self.DATASET_DIR.exists():
            raise FileNotFoundError(f"{self.DATASET_DIR} does not exist!")

        if force or query_bool(f"Delete {self.DATASET_DIR}?", default=False):
            try:  # remove the dataset directory
                shutil.rmtree(self.DATASET_DIR)
            except Exception as exc:
                raise RuntimeError(f"Failed to delete {self.DATASET_DIR}") from exc

            # recreate the dataset directory
            self.DATASET_DIR.mkdir(parents=True, exist_ok=True)
            return

        # else do nothing
        self.LOGGER.debug("Dataset files not deleted.")

    # endregion reset methods ----------------------------------------------------------

    # region properties ----------------------------------------------------------------
    @property
    def version_info(self) -> tuple[int, ...]:
        r"""Version information of the dataset."""
        if self.__version__ is None:
            return ()
        # FIXME: preferably use packaging.version.parse or custom version parser
        return tuple(int(i) for i in self.__version__.split("."))

    @property
    @abstractmethod
    def table_names(self) -> Collection[Key]:  # pyright: ignore[reportRedeclaration]
        r"""READ-ONLY: The names of the tables."""
        # TODO: use abstract-attribute!
        # SEE: https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

    # SEE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020
    table_names: Collection[Key] | cached_property[Collection[Key]]  # type: ignore[no-redef]
    r"""READ-ONLY: The names of the tables."""

    # @property
    # def tables(self) -> dict[Key, T]:
    #     r"""Store cached version of dataset."""
    #     if self._tables is EMPTY_MAP:
    #         # (self.load, (key,), {}) → self.load(key=key) when tables[key] is accessed.
    #         self._tables = LazyDict.from_func(
    #             self.table_names,
    #             self.load,
    #             kwargs={"initializing": True},
    #             type_hint=get_return_typehint(self.clean_table),
    #         )
    #
    #     return self._tables

    @cached_property
    def dataset_files(self) -> dict[Key, str]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.{self.DEFAULT_FILE_FORMAT}" for key in self.table_names}

    @cached_property
    def dataset_paths(self) -> dict[Key, Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return {k: self.DATASET_DIR / fname for k, fname in self.dataset_files.items()}

    @cached_property
    def _enable_key_attributes(self) -> bool:
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
            r"""Test if attribute exists using only __getattribute__ and not __getattr__."""
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

    # endregion properties -------------------------------------------------------------

    # region dunder methods ------------------------------------------------------------
    def __dir__(self) -> list[str]:
        if self._enable_key_attributes:
            return list(super().__dir__()) + list(self.table_names)
        return list(super().__dir__())

    def __getattr__(self, key: Key, /) -> T:  # type: ignore[misc]
        r"""Get attribute."""
        if self._enable_key_attributes and key in self.table_names:
            return self.tables[key]
        return self.__getattribute__(key)

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.tables)

    def __iter__(self) -> Iterator[Key]:
        r"""Return an iterator over the dataset."""
        return iter(self.tables)

    def __getitem__(self, key: Key, /) -> T:
        r"""Return the sample at index `idx`."""
        # need to manually raise KeyError otherwise __getitem__ will execute.
        if key not in self.tables:
            raise KeyError(f"Key {key} not in {self.__class__.__name__}!")
        return self.tables[key]

    def __repr__(self) -> str:
        r"""Pretty Print."""
        return repr_mapping(self.tables, wrapped=self, modifier=self.__version__)

    # endregion dunder methods ---------------------------------------------------------

    # region download methods ----------------------------------------------------------
    def download_file(self, fname: str, /) -> None:
        r"""Download a single rawdata file.

        Override this method for custom download logic.
        """
        if self.SOURCE_URL is NOT_GIVEN:
            self.LOGGER.debug("Dataset provides no base_url. Assumed offline")
            return

        url = self.SOURCE_URL + fname
        path = self.RAWDATA_DIR / fname
        self.LOGGER.debug("Downloading %s from %s", fname, url)
        remote.download(url, path)

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
            for _key in (
                pbar := tqdm(
                    self.rawdata_paths,
                    desc="Downloading files",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_postfix(file=_key)
                self.download(key=_key, force=force, validate=validate)
            return

        # Check if the file already exists.
        if self.rawdata_files_exist(key) and not force:
            self.LOGGER.debug("Files already exist. Skipping download.")
            return

        # Download the file.
        with timer() as t:
            self.download_file(key)
        self.LOGGER.debug("Downloaded file <%s> in %s", key, t.value)

        # Validate the file.
        if validate and self.rawdata_hashes is not EMPTY_MAP:
            self.validate_rawdata(key)

    # endregion download methods -------------------------------------------------------

    # region cleaning methods ----------------------------------------------------------
    @abstractmethod
    def clean_table(self, key: Key) -> Optional[T]:
        r"""Create the cleaned table for the given key.

        If a table is returned, the `self.serialize` method is used to write it to disk.
        If manually writing the table to disk, return None.
        """
        ...

    @final
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
        if (
            validate_rawdata
            and self.rawdata_hashes is not EMPTY_MAP
            and not self.rawdata_valid
        ):
            raise ValueError("Raw data files are not valid!")

        # skip if cleaned files already exist
        if not force and self.dataset_files_exist(key=key):
            self.LOGGER.debug("Table already cleaned, skipping <%s>", key)
            return

        # key=None: Recursively clean all tables
        if key is None:
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
            return

        # Clean the selected table
        with timer() as t:
            df = self.clean_table(key=key)
        self.LOGGER.debug("Cleaned table <%s> in %s", key, t.value)

        if df is not None:
            with timer() as t:
                self.serialize_table(df, self.dataset_paths[key])
            self.LOGGER.info("Serialized table <%s> in %s", key, t.value)

        # Validate the cleaned table
        if validate and self.dataset_hashes is not EMPTY_MAP:
            self.validate_dataset(key)

    # endregion cleaning methods -------------------------------------------------------

    # region loading methods -----------------------------------------------------------
    def load_table(self, *, key: Key) -> T:
        r"""Load the selected table.

        By default, `self.deserialize` is used to load the table from disk.
        Override this method if you want to customize loading the table from disk.
        """
        return self.deserialize_table(self.dataset_paths[key])

    @overload
    def load(
        self,
        key: Key,
        *,
        force: bool = ...,
        validate: bool = ...,
        initializing: bool = ...,
    ) -> T: ...
    @overload
    def load(
        self,
        *,
        force: bool = ...,
        validate: bool = ...,
        initializing: bool = ...,
    ) -> dict[Key, T]: ...
    @final
    def load(
        self,
        key: Optional[Key] = None,
        *,
        force: bool = False,
        validate: bool = True,
        initializing: bool = False,
    ) -> T | dict[Key, T]:
        r"""Load the selected DATASET_OBJECT.

        Args:
            key: The key associated with the datset
            force: Reload the dataset even if it already exists.
            validate: Validate the dataset against hash.
            initializing: Whether this is the first time the dataset is being loaded.

        Returns:
            The loaded dataset
        """
        # key=None: load all tables.
        if key is None:
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
            return self.tables

        # Skip if already loaded.
        if not force and not initializing:
            return self.tables[key]

        # Create the pre-processed dataset file if it doesn't exist.
        if not self.dataset_files_exist(key=key):
            self.clean(key=key, force=force, validate=validate)

        # Validate file if hash is provided.
        if validate and self.dataset_hashes is not EMPTY_MAP:
            self.validate_dataset(key)

        # Load the table, make sure to use the cached version if it exists.
        with timer() as t:
            table = self.load_table(key=key)
        self.LOGGER.info("Loaded table <%s> in %s", key, t.value)

        # Validate the loaded table.
        # FIXME: infinite recursion
        # if validate and self.table_hashes is not NotImplemented:
        # self.validate_tables(key)

        return table

    # endregion loading methods --------------------------------------------------------

    # region validation methods --------------------------------------------------------
    @cached_property
    def rawdata_valid(self) -> bool:
        r"""Check if raw data files exist."""
        return self.validate_rawdata()

    @cached_property
    def dataset_valid(self) -> bool:
        r"""Check if dataset files exist."""
        return self.validate_dataset()

    @cached_property
    def tables_valid(self) -> bool:
        r"""Check if tables are valid."""
        return self.validate_tables()

    def rawdata_files_exist(self, key: Optional[str] = None) -> bool:
        r"""Check if raw data files exist."""
        if key is None:
            return paths_exists(self.rawdata_paths)
        if key not in self.rawdata_paths:
            raise KeyError(f"{key=} not in {self.rawdata_paths=}")
        return paths_exists(self.rawdata_paths[key])

    def dataset_files_exist(self, key: Optional[Key] = None) -> bool:
        r"""Check if dataset files exist."""
        if key is None:
            return paths_exists(self.dataset_paths)
        if key not in self.dataset_paths:
            raise KeyError(f"{key=} not in {self.dataset_paths=}")
        return paths_exists(self.dataset_paths[key])

    def validate_rawdata(
        self, key: Optional[str] = None, errors: ErrorHandler.Mode = "raise"
    ) -> bool:
        r"""Validate the rawdata files."""
        if key is None:
            self.LOGGER.debug("Validating raw data files.")
            result = True
            exceptions = []
            for _key in (
                pbar := tqdm(
                    self.rawdata_paths,
                    desc="Validating files",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_postfix(file=_key)
                try:
                    result &= self.validate_rawdata(key=_key, errors="raise")
                except ValidationError as exc:
                    result = False
                    exceptions.append(exc)
            if exceptions:
                failed = "\n".join(str(exc) for exc in exceptions)
                ErrorHandler(errors).emit(f"Some tables failed validation:\n{failed}")
            return result

        self.LOGGER.debug("Validating %s.", key)
        return validate_file(
            self.rawdata_paths[key],
            self.rawdata_hashes.get(key),
            errors=errors,
        )

    def validate_dataset(
        self, key: Optional[Key] = None, *, errors: ErrorHandler.Mode = "warn"
    ) -> bool:
        r"""Validate the dataset."""
        if key is None:
            self.LOGGER.debug("Validating dataset.")
            result = True
            exceptions = []
            for _key in (
                pbar := tqdm(
                    self.table_names,
                    desc="Validating dataset",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_postfix(table=_key)
                try:
                    result &= self.validate_dataset(key=_key, errors="raise")
                except ValidationError as exc:
                    result = False
                    exceptions.append(exc)
            if exceptions:
                failed = "\n".join(str(exc) for exc in exceptions)
                ErrorHandler(errors).emit(f"Some tables failed validation:\n{failed}")
            return result

        self.LOGGER.debug("Validating %s.", key)
        return validate_file(
            self.dataset_paths[key],
            self.dataset_hashes.get(key),
            errors=errors,
        )

    def validate_tables(
        self, key: Optional[Key] = None, *, errors: ErrorHandler.Mode = "warn"
    ) -> bool:
        if key is None:
            self.LOGGER.debug("Validating tables.")
            result = True
            exceptions = []
            for _key in (
                pbar := tqdm(
                    self.table_names,
                    desc="Validating tables",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_postfix(table=_key)
                try:
                    result &= self.validate_tables(key=_key, errors="raise")
                except ValidationError as exc:
                    result = False
                    exceptions.append(exc)
            if exceptions:
                failed = "\n".join(str(exc) for exc in exceptions)
                ErrorHandler(errors).emit(f"Some tables failed validation:\n{failed}")
            return result

        self.LOGGER.debug("Validating %s.", key)
        return validate_table(
            self.tables[key],
            self.table_hashes.get(key),
            errors=errors,
        )

    # endregion validation methods -----------------------------------------------------
