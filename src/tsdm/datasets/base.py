r"""Base Classes for dataset."""

# NOTE: signature of metaclass.__init__ should match that of type.__new__
# NOTE: type.__new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> Self
# NOTE: type.__init__(self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> None

__all__ = [
    # Constants
    "VERSION_REGEX",
    # ABCs & Protocols
    "Dataset",
    "DatasetBase",
    "DatasetMeta",
    # Functions
    "check_version_format",
    "parse_version",
]
import logging
import re
import shutil
import warnings
import webbrowser
from abc import abstractmethod
from collections.abc import Collection, Iterator, Mapping
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
from tsdm.constants import EMPTY_MAP, UNDEFINED
from tsdm.datatools import serialize
from tsdm.pprint import repr_mapping
from tsdm.testing.validation import (
    ErrorHandler,
    ValidationError,
    validate_file_hash,
    validate_table_hash,
    validate_table_schema,
    validate_table_shape,
)
from tsdm.types.aliases import FilePath
from tsdm.utils import nested_paths_exist, remote
from tsdm.utils._utils import query_bool
from tsdm.utils.contextmanagers import timer
from tsdm.utils.funcutils import get_return_typehint
from tsdm.utils.lazydict import LazyDict


@runtime_checkable
class Dataset[Key, T](Protocol):  # +T
    r"""Protocol for Dataset.

    A dataset is a collection of table-like objects indexed by keys.

    This protocol describes a reduced interface of the `DatasetBase` class,
    and covers only methods that should be used at call sites, that is, methods
    from an already instantiated object.
    """

    @property
    @abstractmethod
    def tables(self) -> Mapping[Key, T]:
        r"""READ-ONLY: The tables that make up the dataset."""

    @property
    @abstractmethod
    def table_names(self) -> Collection[Key]:
        r"""READ-ONLY: The names of the tables that make up the dataset."""

    @classmethod
    def deserialize(cls, filepath: FilePath, /) -> Self: ...
    def serialize(self, filepath: FilePath, /) -> None: ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Key]: ...
    def __getitem__(self, key: Key, /) -> T: ...
    def __contains__(self, key: object, /) -> bool: ...


class DatasetMeta(ProtocolMeta):
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
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        assert isinstance(cls.LOGGER, logging.Logger)

        if "ID" not in namespace:
            cls.ID: str = cls.__qualname__
        assert isinstance(cls.ID, str)

        if "ROOT_DIR" not in namespace:
            cls.ROOT_DIR: Path = CONFIG.DATASET_DIR / cls.ID
        assert isinstance(cls.ROOT_DIR, Path)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: N805
        r"""When an instance of the class is created, this method is called."""
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class DatasetBase[Key: str, T](
    Mapping[Key, T], Dataset[Key, T], metaclass=DatasetMeta
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
    SOURCE_URL: ClassVar[str] = UNDEFINED
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL: ClassVar[Optional[str]] = None
    r"""HTTP address containing additional information about the dataset."""
    ID: ClassVar[str]  # typically the class name
    r"""READ_ONLY: Unique identifier for the dataset."""
    ROOT_DIR: ClassVar[Path]  # on class:  ~/.tsdm/datasets/<name>/
    r"""Location where the dataset is stored."""
    # endregion class attributes -------------------------------------------------------

    # region abstract members ----------------------------------------------------------
    @property
    @abstractmethod
    def rawdata_files(self) -> Collection[str]: ...  # pyright: ignore[reportRedeclaration]
    @property
    @abstractmethod
    def table_names(self) -> Collection[Key]: ...  # pyright: ignore[reportRedeclaration]

    rawdata_files: Collection[str]  # type: ignore[no-redef]
    r"""READ_ONLY: The names of the raw data files that make up the dataset."""
    table_names: Collection[Key]  # type: ignore[no-redef]  # pyright: ignore[reportIncompatibleMethodOverride]
    r"""READ_ONLY: The names of the tables that make up the dataset."""
    # endregion abstract members  ------------------------------------------------------

    # region derived members -----------------------------------------------------------
    STORAGE_DIR: Path  # typically ~/.tsdm/datasets/<name>/<version>/
    r"""READ_ONLY: Location where the dataset version is stored."""
    RAWDATA_DIR: Path  # typically ~/.tsdm/datasets/<name>/<version>/raw
    r"""READ_ONLY: Location where the raw data is stored."""
    DATASET_DIR: Path  # typically ~/.tsdm/datasets/<name>/<version>/processed
    r"""READ_ONLY:Location where the processed data is stored."""
    METADATA_DIR: Path  # typically ~/.tsdm/datasets/<name>/<version>/meta
    r"""READ_ONLY:Location where the metadata is stored."""
    # endregion derived members --------------------------------------------------------

    # region instance attributes -------------------------------------------------------
    # TODO: Use typing.ReadOnly (https://peps.python.org/pep-0767/)
    __version__: str | None = None
    r"""READ-ONLY: The version of the dataset (None=unversioned)."""
    rawdata_hashes: Mapping[str, str | None] = EMPTY_MAP
    r"""READ-ONLY: Hashes of the raw dataset file(s)."""
    rawdata_schemas: Mapping[str, Mapping[str, str]] = EMPTY_MAP
    r"""READ-ONLY: Schemas for the raw dataset tables(s)."""
    rawdata_shapes: Mapping[str, tuple[int, ...]] = EMPTY_MAP
    r"""READ-ONLY: Shapes for the raw dataset tables(s)."""

    # FIXME: Hack due to lack of ReadOnly attributes.
    # We can't use Key, as that would screw up covariance.
    type _Key = str
    r"""Type alias for the key of the dataset."""
    dataset_hashes: Mapping[Key, str | None] = EMPTY_MAP
    r"""READ-ONLY: Hashes of the cleaned dataset file(s)."""
    table_hashes: Mapping[_Key, str | None] = EMPTY_MAP
    r"""READ-ONLY: Hashes of the in-memory cleaned dataset table(s)."""
    table_schemas: Mapping[_Key, Mapping[str, str]] = EMPTY_MAP
    r"""READ-ONLY: Schemas of the in-memory cleaned dataset table(s)."""
    table_shapes: Mapping[_Key, tuple[int, ...]] = EMPTY_MAP
    r"""READ-ONLY: Shapes of the in-memory cleaned dataset table(s)."""
    # endregion instance attributes ----------------------------------------------------

    @cached_property
    def tables(self) -> LazyDict[Key, T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return LazyDict.from_func(
            self.table_names,
            self.load,
            kwargs={"initializing": True},
            type_hint=get_return_typehint(self.clean_table),
        )

    # region constructors --------------------------------------------------------------
    @classmethod
    def from_tables(cls, tables: Mapping[Key, T], /) -> Self:
        r"""Create a dataset from a table."""
        obj = cls(initialize=False)

        # set the tables, raises KeyError if any key is missing
        for key in obj:
            obj.tables[key] = tables[key]

        # check for superfluous keys
        if superfluous_keys := tables.keys() - obj.keys():
            warnings.warn(
                f"Keys {superfluous_keys} are not valid keys for {cls.ID}!"
                f" They will not be added to the dataset.",
                RuntimeWarning,
                stacklevel=2,
            )
        return obj

    @classmethod
    def get_storage_paths(cls, version: str | None, /) -> dict[str, Path]:
        r"""Get the storage paths for the given version."""
        version = cls.check_version(version)
        root_dir = cls.ROOT_DIR / (version or "")
        return {
            CONFIG.DATASET_PATHS.ROOT: root_dir,
            CONFIG.DATASET_PATHS.RAWDATA: root_dir / CONFIG.DATASET_PATHS.RAWDATA,
            CONFIG.DATASET_PATHS.PROCESSED: root_dir / CONFIG.DATASET_PATHS.PROCESSED,
            CONFIG.DATASET_PATHS.METADATA: root_dir / CONFIG.DATASET_PATHS.METADATA,
        }

    @classmethod
    def check_version(cls, version: str | None, /) -> str | None:
        r"""Get the version of the dataset."""
        if version is None:
            return cls.__version__

        version = str(version)
        check_version_format(version)
        return version

    def set_storage_paths(self, version: str | None, /) -> None:
        r"""Set the storage paths for the given version."""
        storage_paths = self.get_storage_paths(version)
        self.ROOT_DIR = storage_paths[CONFIG.DATASET_PATHS.ROOT]  # type: ignore[misc]
        self.RAWDATA_DIR = storage_paths[CONFIG.DATASET_PATHS.RAWDATA]
        self.DATASET_DIR = storage_paths[CONFIG.DATASET_PATHS.PROCESSED]
        self.METADATA_DIR = storage_paths[CONFIG.DATASET_PATHS.METADATA]
        for path in storage_paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def __init__(
        self,
        *,
        initialize: bool = True,
        verbose: bool = True,
        version: Optional[str] = None,
    ) -> None:
        r"""Initialize the dataset.

        Args:
            initialize: Whether to initialize the dataset.
            version: Version of the dataset. Leave empty for unversioned dataset.
            verbose: Whether to print verbose output.
        """
        self.verbose = verbose
        self.initialize = initialize
        self.__version__ = self.check_version(version)
        self.set_storage_paths(version)

    def __post_init__(self) -> None:
        r"""Initialize the dataset."""
        if self.initialize:
            # NOTE: We call clean first for memory efficiency.
            #  Preprocessing can take lots of resources, so we should only load tables
            #  on a need-to-know basis.
            self.clean()
            self.load(initializing=True)

    # endregion constructors -----------------------------------------------------------

    # region classmethods --------------------------------------------------------------
    @classmethod
    def info(cls) -> None:
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            raise NotImplementedError("No INFO_URL provided for this dataset!")
        webbrowser.open_new_tab(cls.INFO_URL)

    @classmethod
    def reset(cls, *, version: Optional[str] = None, force: bool = False) -> None:
        r"""Reset the data folders."""
        cls.reset_rawdata_files(version=version, force=force)
        cls.reset_dataset_files(version=version, force=force)

    @classmethod
    def reset_rawdata_files(
        cls, *, version: Optional[str] = None, force: bool = False
    ) -> None:
        r"""Recreate the rawdata directory."""
        rawdata_dir = cls.RAWDATA_DIR / (version or "")

        if not rawdata_dir.exists():
            raise FileNotFoundError(f"{rawdata_dir} does not exist!")

        if force or query_bool(f"Delete {rawdata_dir}?", default=False):
            try:  # remove the rawdata directory
                shutil.rmtree(rawdata_dir)
            except Exception as exc:
                raise RuntimeError(f"Failed to delete {rawdata_dir}") from exc

            # recreate the rawdata directory
            rawdata_dir.mkdir(parents=True, exist_ok=True)
            return

        # else do nothing
        cls.LOGGER.debug("Rawdata files not deleted.")

    @classmethod
    def reset_dataset_files(
        cls, *, version: Optional[str] = None, force: bool = False
    ) -> None:
        r"""Recreate the dataset directory."""
        dataset_dir = cls.DATASET_DIR / (version or "")

        if not dataset_dir.exists():
            raise FileNotFoundError(f"{dataset_dir} does not exist!")

        if force or query_bool(f"Delete {dataset_dir}?", default=False):
            try:  # remove the dataset directory
                shutil.rmtree(dataset_dir)
            except Exception as exc:
                raise RuntimeError(f"Failed to delete {dataset_dir}") from exc

            # recreate the dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)
            return

        # else do nothing
        cls.LOGGER.debug("Dataset files not deleted.")

    # endregion classmethods -----------------------------------------------------------

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
                    name = cast("Key", Path(fname).stem)
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

    # region properties ----------------------------------------------------------------
    @property
    def version_info(self) -> tuple[int, ...]:
        r"""Version information of the dataset."""
        return parse_version(self.__version__)

    @cached_property
    def rawdata_paths(self) -> Mapping[str, Path]:
        r"""Return mapping from filenames to paths to the rawdata files."""
        return {
            str(fname): (self.RAWDATA_DIR / fname).absolute()
            for fname in self.rawdata_files
        }

    @cached_property
    def dataset_paths(self) -> dict[Key, Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return {
            key: self.DATASET_DIR / f"{key}.{self.DEFAULT_FILE_FORMAT}"
            for key in self.table_names
        }

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

        def attr_exists(obj: object, key: str, /) -> bool:
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
        r"""Dynamically add table names to dir()."""
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
            raise KeyError(f"Key {key} not a member of {self.ID}!")
        return self.tables[key]

    def __repr__(self) -> str:
        r"""Pretty Print."""
        return repr_mapping(self.tables, wrapped=self, modifier=self.__version__)

    # endregion dunder methods ---------------------------------------------------------

    # region download mechanism --------------------------------------------------------
    def download_file(self, fname: str, /) -> None:
        r"""Download a single rawdata file.

        Override this method for custom download logic.
        """
        if self.SOURCE_URL is UNDEFINED:
            self.LOGGER.debug("Dataset provides no base_url. Assumed offline")
            return

        url = self.SOURCE_URL
        if url.endswith("/"):
            url = url + fname
        elif len(self.rawdata_paths) > 1:
            msg = "URL must end with '/' if multiple files are to be downloaded!"
            raise ValueError(msg)

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
            for name in (
                pbar := tqdm(
                    self.rawdata_paths,
                    desc="Downloading files",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_description(f"Downloading file {name!r}")
                self.download(key=name, force=force, validate=validate)
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

    # endregion download mechanism -----------------------------------------------------

    # region cleaning mechanism --------------------------------------------------------
    def clean_table(self, key: Key, /) -> Optional[T]:
        r"""Create the cleaned table for the given key.

        By default, this method redirects to `self.clean_{key}` method.
        Subclasses can either implement these, or override this method directly.

        If a table is returned, the `self.serialize` method is used to write it to disk.
        If manually writing the table to disk, return None.
        """
        if key not in self.table_names:
            raise KeyError(f"Key {key} unknown! Must be one of {self.table_names}")
        try:
            cleaner = getattr(self, f"clean_{key}")
        except AttributeError:
            raise NotImplementedError(
                f"Cleaning method for {key} not implemented!"
                f" Either implement `clean_{key}` or override `clean_table`."
            ) from None
        else:
            return cleaner()

    @final
    def clean(
        self,
        key: Optional[Key] = None,
        /,
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
        if not force and self.dataset_files_exist(key):
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
                pbar.set_description(f"Cleaning table {name!r}")
                self.clean(
                    name,
                    force=force,
                    validate=validate,
                    validate_rawdata=False,
                )
            return

        # Clean the selected table
        with timer() as t:
            df = self.clean_table(key)
        self.LOGGER.debug("Cleaned table <%s> in %s", key, t.value)

        if df is not None:
            with timer() as t:
                self.serialize_table(df, self.dataset_paths[key])
            self.LOGGER.info("Serialized table <%s> in %s", key, t.value)

        # Validate the cleaned table
        if validate and self.dataset_hashes is not EMPTY_MAP:
            self.validate_dataset(key)

    # endregion cleaning mechanism -----------------------------------------------------

    # region loading mechanism ---------------------------------------------------------
    def load_table(self, key: Key, /) -> T:
        r"""Load the selected table.

        By default, `self.deserialize` is used to load the table from disk.
        Override this method if you want to customize loading the table from disk.
        """
        return self.deserialize_table(self.dataset_paths[key])

    @overload
    def load(
        self,
        key: Key,
        /,
        *,
        force: bool = ...,
        validate: bool = ...,
        initializing: bool = ...,
    ) -> T: ...
    @overload
    def load(
        self,
        /,
        *,
        force: bool = ...,
        validate: bool = ...,
        initializing: bool = ...,
    ) -> dict[Key, T]: ...
    @final
    def load(
        self,
        key: Optional[Key] = None,
        /,
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
                pbar.set_description(f"Loading table {name!r}")
                self.load(name, force=force, validate=validate)
            return self.tables

        # Skip if already loaded.
        if not force and not initializing:
            return self.tables[key]

        # Create the pre-processed dataset file if it doesn't exist.
        if not self.dataset_files_exist(key):
            self.clean(key, force=force, validate=validate)

        # Validate file if hash is provided.
        if validate and self.dataset_hashes is not EMPTY_MAP:
            self.validate_dataset(key)

        # Load the table, make sure to use the cached version if it exists.
        with timer() as t:
            table = self.load_table(key)
        self.LOGGER.info("Loaded table <%s> in %s", key, t.value)

        # Validate the loaded table.
        # FIXME: infinite recursion

        return table

    # endregion loading mechanism ------------------------------------------------------

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

    def rawdata_files_exist(self, key: Optional[str] = None, /) -> bool:
        r"""Check if raw data files exist."""
        if key is None:
            return nested_paths_exist(self.rawdata_paths)
        if key not in self.rawdata_paths:
            raise KeyError(f"{key=} not in {self.rawdata_paths=}")
        return nested_paths_exist(self.rawdata_paths[key])

    def dataset_files_exist(self, key: Optional[Key] = None, /) -> bool:
        r"""Check if dataset files exist."""
        if key is None:
            return nested_paths_exist(self.dataset_paths)
        if key not in self.dataset_paths:
            raise KeyError(f"{key=} not in {self.dataset_paths=}")
        return nested_paths_exist(self.dataset_paths[key])

    def validate_rawdata(
        self, key: Optional[str] = None, /, *, errors: ErrorHandler.Mode = "raise"
    ) -> bool:
        r"""Validate the rawdata files."""
        if key is None:
            self.LOGGER.debug("Validating raw data files.")
            result = True
            exceptions: dict[str, ValidationError] = {}
            for name in (
                pbar := tqdm(
                    self.rawdata_paths,
                    desc="Validating files",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_description(f"Validating file {name!r}")
                try:
                    result &= self.validate_rawdata(name, errors="raise")
                except ValidationError as exc:
                    result = False
                    exceptions[name] = exc
            if exceptions:
                failed = "\n".join(f"{key}: {exc}" for key, exc in exceptions.items())
                msg = f"Some raw data files failed validation:\n{failed}"
                ErrorHandler(errors).emit(msg)
            return result

        self.LOGGER.debug("Validating %s.", key)
        return validate_file_hash(
            self.rawdata_paths[key],
            self.rawdata_hashes.get(key),
            errors=errors,
        )

    def validate_dataset(
        self, key: Optional[Key] = None, /, *, errors: ErrorHandler.Mode = "warn"
    ) -> bool:
        r"""Validate the dataset."""
        if key is None:
            self.LOGGER.debug("Validating dataset.")
            result = True
            exceptions: dict[str, ValidationError] = {}
            for name in (
                pbar := tqdm(
                    self.table_names,
                    desc="Validating dataset",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_description(f"Validating table {name!r}")
                try:
                    result &= self.validate_dataset(name, errors="raise")
                except ValidationError as exc:
                    result = False
                    exceptions[name] = exc
            if exceptions:
                failed = "\n".join(f"{key}: {exc}" for key, exc in exceptions.items())
                ErrorHandler(errors).emit(f"Some tables failed validation:\n{failed}")
            return result

        self.LOGGER.debug("Validating %s.", key)
        return validate_file_hash(
            self.dataset_paths[key],
            self.dataset_hashes.get(key),
            errors=errors,
        )

    def validate_tables(
        self, key: Optional[Key] = None, /, *, errors: ErrorHandler.Mode = "warn"
    ) -> bool:
        if key is None:
            self.LOGGER.debug("Validating tables.")
            result = True
            exceptions: dict[str, ValidationError] = {}
            for name in (
                pbar := tqdm(
                    self.table_names,
                    desc="Validating tables",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                pbar.set_description(f"Validating table {name!r}")
                try:
                    result &= self.validate_tables(name, errors="raise")
                except ValidationError as exc:
                    result = False
                    exceptions[name] = exc
            if exceptions:
                failed = "\n".join(f"{key}: {exc}" for key, exc in exceptions.items())
                ErrorHandler(errors).emit(f"Some tables failed validation:\n{failed}")
            return result

        excs: list[ValidationError] = []
        self.LOGGER.debug(f"{key=} Validating table shape")
        try:
            shapes_match = validate_table_shape(
                self.tables[key],  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
                expected_shape=self.table_shapes.get(key),
                errors=errors,
            )
        except ValidationError as exc1:
            shapes_match = False
            excs.append(exc1)

        self.LOGGER.debug(f"{key=} Validating table schema")
        try:
            schema_matches = validate_table_schema(
                self.tables[key],
                expected_shema=self.table_schemas.get(key),
                errors=errors,
            )
        except ValidationError as exc2:
            schema_matches = False
            excs.append(exc2)

        self.LOGGER.debug(f"{key=} Validating table hash")
        try:
            hash_matches = validate_table_hash(
                self.tables[key],
                expected_hash=self.table_hashes.get(key),
                skipif_no_reference=True,
                errors=errors,
            )
        except ValidationError as exc3:
            hash_matches = False
            excs.append(exc3)

        if excs:
            raise ValidationError from ExceptionGroup("Table validation failed", excs)

        return shapes_match and schema_matches and hash_matches

    # endregion validation methods -----------------------------------------------------


VERSION_REGEX = r"\d+(?:\.\d+)*"
r"""Regex for version strings."""


def check_version_format(version: str | None, /) -> None:
    r"""Check if version string is valid."""
    if version is not None and not re.fullmatch(VERSION_REGEX, version):
        raise ValueError(
            f"Version {version!r} is not valid! "
            "Version must be of the form 'X.Y.Z' or 'latest'."
        )


def parse_version(version: str | None, /) -> tuple[int, ...]:
    r"""Parse version string into a tuple of integers."""
    check_version_format(version)
    if version is None:
        return ()
    return tuple(int(part) for part in version.split("."))
