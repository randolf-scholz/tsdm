r"""Validation utilities for hash and schema validation."""

__all__ = [
    # classes
    "ErrorHandler",
    "ValidationError",
    # functions
    "make_error_handler",
    "validate_file_hash",
    "validate_hash",
    "validate_table_hash",
    "validate_table_schema",
    "validate_table_shape",
]

import hashlib
import logging
import warnings
from collections.abc import Mapping, Sequence
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, assert_never, overload

import polars as pl
import pyarrow as pa
from pandas import DataFrame, Index, MultiIndex, Series

from tsdm.config import CONFIG
from tsdm.types.aliases import FilePath, FileStream
from tsdm.types.extra import SupportsShape

from .hashutils import Hash, hash_array, hash_file


class ValidationError(ValueError):
    r"""Validation error for hash validation."""


class ErrorHandler:
    r"""Validation mode for hash validation."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")

    class MODE(StrEnum):
        r"""Validation mode for hash validation."""

        IGNORE = "ignore"
        LOG = "log"
        WARN = "warn"
        RAISE = "raise"

    # type alias for handling both enums and string literals.
    # SEE: https://discuss.python.org/t/amend-pep-586-to-make-enum-values-subtypes-of-literal/59456
    type Mode = "MODE" | Literal["ignore", "log", "warn", "raise"]

    mode: MODE
    r"""Validation mode for hash validation."""

    def __init__(
        self,
        mode: Mode,
        /,
        *,
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        self.mode = self.MODE(mode)
        self.prefix = prefix
        self.postfix = postfix

    def emit(self, msg: str, /, *, valid: bool = False) -> None:
        r"""Emit a validation message according to the selected mode."""
        msg = f"{self.prefix}{msg}{self.postfix}"

        match self.mode:
            case self.MODE.RAISE:
                if not valid:
                    raise ValidationError(msg)
            case self.MODE.WARN:
                if not valid:
                    warnings.warn(msg, UserWarning, stacklevel=2)
            case self.MODE.LOG:
                self.LOGGER.info(msg)
            case self.MODE.IGNORE:
                pass
            case other:
                assert_never(other)


@overload
def make_error_handler(arg: ErrorHandler, /) -> ErrorHandler: ...
@overload
def make_error_handler(
    arg: ErrorHandler.Mode, /, *, prefix: str = ..., postfix: str = ...
) -> ErrorHandler: ...
def make_error_handler(
    arg: ErrorHandler | ErrorHandler.Mode, /, *, prefix: str = "", postfix: str = ""
) -> ErrorHandler:
    r"""Create a new `ErrorHandler` from an existing one or from a mode."""
    match arg:
        case ErrorHandler() as handler:
            return handler
        case mode:
            return ErrorHandler(mode, prefix=prefix, postfix=postfix)


def validate_hash(
    hash_value: Hash | str | None,
    /,
    expected_hash: Hash | str | None,
    *,
    errors: ErrorHandler | ErrorHandler.Mode = "raise",
) -> bool:
    r"""Compare a hash value against a reference hash value.

    Args:
        hash_value: The hash value to validate.
        expected_hash: The reference hash value.
        errors: How to handle errors, one of "ignore", "log", "warn", "raise".

    Returns:
        bool: True if the hash matches the reference hash, False otherwise.

    Raises:
        ValidationError: If errors="raise" and the hash does not match the reference hash.
    """
    error_handler = make_error_handler(errors)
    actual_hash = Hash.from_value(hash_value)
    expected_hash = Hash.from_value(expected_hash)

    match actual_hash, expected_hash:
        case None, None:
            msg = "⚠️  No hashes given, skipping validation."
            hashes_match = True
        case None, _:
            msg = "⚠️  No actual hash given, skipping validation."
            hashes_match = False
        case _, None:
            msg = "⚠️  No reference hash given, skipping validation."
            hashes_match = False
        case _, _:
            assert actual_hash is not None  # needed for type checker
            assert expected_hash is not None  # needed for type checker
            algs_match = actual_hash.hash_algorithm == expected_hash.hash_algorithm
            hashes_match = actual_hash.hash_value == expected_hash.hash_value

            if not algs_match:
                msg = (
                    f"Hash algorithm mismatch:"
                    f" {actual_hash.hash_algorithm!r} ≠ {expected_hash.hash_algorithm!r}."
                )
            elif not hashes_match:
                msg = (
                    f"Hash mismatch: ❌️"
                    f"\n\texpected: {expected_hash!s}"
                    f"\n\tactual  : {actual_hash!s}"
                )
            else:
                msg = (
                    f"Hashes match! ✅️"
                    f"\n\texpected: {expected_hash!s}"
                    f"\n\tactual  : {actual_hash!s}"
                )

    error_handler.emit(msg, valid=hashes_match)
    return hashes_match


def validate_file_hash(
    path_or_stream: FilePath | FileStream,
    /,
    expected_hash: Hash | str | None,
    *,
    hash_algorithm: Optional[str] = None,
    errors: ErrorHandler.Mode = "raise",
    skipif_no_reference: bool = False,
) -> bool:
    r"""Validate file(s), given reference hash value(s).

    Args:
        path_or_stream: The file or binary stream to validate. Streams are hashed from the
            beginning and returned to their original position afterwards.
        expected_hash: The reference hash value.
        hash_algorithm: The hash algorithm to use.
        errors: How to handle errors. Can be "warn", "raise", "log" or "ignore". (default: "warn")
        skipif_no_reference: If True, skip validation if no reference hash is given.

    Returns:
        bool: True if the hash matches the reference hash, False otherwise.

    Raises:
        ValidationError: If errors="raise" and the table hash does not match the reference hash.
    """
    # region input validation ----------------------------------------------------------
    match path_or_stream:
        case str() | PathLike():
            file = Path(path_or_stream)
            error_handler = make_error_handler(errors, prefix=f"{file!s}: ")
        case stream:
            file = None
            error_handler = make_error_handler(
                errors, prefix=f"<{type(stream).__name__}>: "
            )
    expected_hash = Hash.from_value(expected_hash)

    if expected_hash is None and skipif_no_reference:
        return True

    if file is not None and file.suffix == ".parquet":
        msg = f"{file!s}: ⚠️ refusing to hash, parquet is not binary stable!"
        error_handler.emit(msg)
        return False

    # Determine the hash algorithm to use.
    match hash_algorithm:
        case str(hash_alg):
            pass
        case None:
            match expected_hash:
                case Hash(hash_algorithm=str(hash_alg)):
                    pass  # assume same algorithm as reference
                case _:
                    warnings.warn(
                        "No hash algorithm given for reference hash!"
                        f"Using {CONFIG.DEFAULT_HASH_METHOD!r} as default.",
                        UserWarning,
                        stacklevel=2,
                    )
                    hash_alg = CONFIG.DEFAULT_HASH_METHOD
        case _:
            raise TypeError(f"Invalid hash algorithm type: {type(hash_algorithm)}")

    # Compute the hash
    match path_or_stream:
        case str() | PathLike():
            actual_hash = hash_file(path_or_stream, hash_alg)
        case stream:
            position = stream.tell()
            hasher = hashlib.new(hash_alg)
            try:
                stream.seek(0)
                for byte_block in iter(lambda: stream.read(65536), b""):
                    hasher.update(byte_block)
            finally:
                stream.seek(position)
            actual_hash = Hash(hasher.hexdigest(), hasher.name)
    return validate_hash(actual_hash, expected_hash, errors=error_handler)


def validate_table_hash(
    table: Any,
    /,
    expected_hash: Hash | str | None,
    *,
    skipif_no_reference: bool = False,
    errors: ErrorHandler.Mode = "warn",
    hash_algorithm: Optional[str] = None,
) -> bool:
    r"""Validate the hash of a table-like object, given a reference hash value.

    Args:
        table: The table to validate.
        expected_hash: The reference hash value.
        hash_algorithm: The hash algorithm to use.
        errors: How to handle errors, one of "ignore", "log", "warn", "raise".
        skipif_no_reference: If True, skip validation if no reference hash is given.

    Returns:
        bool: True if the hash matches the reference hash, False otherwise.

    Raises:
        ValidationError: If errors="raise" and the table hash does not match the reference hash.
    """
    # Try to determine the hash algorithm from the array type
    cls = type(table)
    name = f"{cls} of shape={table.shape}"
    expected_hash = Hash.from_value(expected_hash)
    error_handler = make_error_handler(errors, prefix=f"{name!s}: ")

    if expected_hash is None and skipif_no_reference:
        return True

    # Determine the hash algorithm
    match hash_algorithm:
        case str(hash_alg):
            pass
        case None:
            match expected_hash:
                case Hash(hash_algorithm=str(hash_alg)):
                    pass  # assume same algorithm as reference
                case _:
                    warnings.warn(
                        "No hash algorithm given for reference hash!"
                        f"Using {CONFIG.DEFAULT_HASH_METHOD!r} as default.",
                        UserWarning,
                        stacklevel=2,
                    )
                    hash_alg = CONFIG.DEFAULT_HASH_METHOD
        case _:
            raise TypeError(f"Invalid hash algorithm type: {type(hash_algorithm)}")

    # Compute the hash.
    actual_hash = hash_array(table, hash_alg)
    return validate_hash(actual_hash, expected_hash, errors=error_handler)


def validate_table_shape(
    table: SupportsShape,
    /,
    expected_shape: tuple[int, ...] | None,
    *,
    errors: ErrorHandler.Mode = "warn",
) -> bool:
    r"""Validate the shape of a table-like object, given a reference shape value."""
    error_handler = make_error_handler(errors)

    match table:
        case SupportsShape(shape=actual_shape):
            pass
        case _:
            raise TypeError(f"Cannot get shape of object of type {type(table)}!")

    # Validate shape.
    match actual_shape, expected_shape:
        case _, None:
            msg_shape = f"No reference shape given, {actual_shape}."
            shapes_match = True
        case _, _:
            if shapes_match := actual_shape == expected_shape:
                msg_shape = "Table shape validated successfully."
            else:
                msg_shape = (
                    f"Table {actual_shape=!r} does not match {expected_shape=!r}!"
                )
    error_handler.emit(msg_shape, valid=shapes_match)
    return shapes_match


def validate_table_schema(
    table: Any,
    /,
    *,
    expected_schema: Sequence[str] | Mapping[str, Any] | pa.Schema | None,
    errors: ErrorHandler.Mode = "warn",
) -> bool:
    r"""Validate the schema of a `pandas` object, given schema values from a table.

    Args:
        table: The table to validate.
        expected_schema: Checks if the columns and dtypes of the table match the reference schema.
        errors: How to handle errors, one of "ignore", "log", "warn", "raise".

    Returns:
        bool: True if the schema matches the reference schema, False otherwise.

    Raises:
        ValidationError: If errors="raise" and the table schema does not match the reference schema.
    """
    error_handler = make_error_handler(errors)

    actual_columns: Sequence | None
    expected_columns: Sequence | None
    actual_dtypes: Mapping | None
    expected_dtypes: Mapping | None
    index_columns: Sequence

    # get data shape, columns and dtypes from table
    match table:
        case MultiIndex(names=names, dtypes=dtypes):
            actual_columns = names
            actual_dtypes = dict(zip(names, dtypes, strict=True))
            index_columns = []
        case Index() as index:
            actual_columns = [index.name]
            actual_dtypes = {index.name: index.dtype}
            index_columns = []
        case Series() as series:
            actual_columns = [series.name]
            actual_dtypes = {series.name: series.dtype}
            index_columns = series.index.names
        case DataFrame() as df:
            actual_columns = df.columns.tolist()
            actual_dtypes = df.dtypes.to_dict()
            index_columns = df.index.names
        case pa.Table(schema=schema):
            actual_columns = schema.names
            actual_dtypes = dict(zip(schema.names, schema.types, strict=True))
            index_columns = []
        case pl.DataFrame() as df:
            actual_columns = df.columns
            actual_dtypes = {col: df[col].dtype for col in df.columns}
            index_columns = []
        case _:
            raise NotImplementedError(
                f"Cannot validate schema for {type(table)} objects!"
            )

    # get reference columns and dtypes
    match expected_schema:
        case pa.Schema() as schema:
            expected_columns = schema.names
            expected_dtypes = dict(zip(schema.names, schema.types, strict=True))
        case Mapping() as mapping:
            expected_columns = list(mapping.keys())
            expected_dtypes = mapping
        case Sequence() as seq:
            expected_columns = seq
            expected_dtypes = dict.fromkeys(seq)
        case None:
            expected_columns = None
            expected_dtypes = None
        case _:
            raise TypeError(f"Invalid reference schema type! {type(expected_schema)=}")

    # Validate columns.
    match actual_columns, expected_columns:
        case None, None:
            msg = "No columns in actual table and no reference columns given, skipping column validation."
            columns_match = True
        case _, None:
            msg = "No reference columns given, skipping column validation."
            columns_match = False
        case None, _:
            msg = "No columns in actual table, but reference columns given!"
            columns_match = False
        case _, _:
            assert actual_columns is not None  # needed for type checker
            assert expected_columns is not None  # needed for type checker
            missing_columns = set(expected_columns) - set(actual_columns)
            missing_columns = missing_columns - set(index_columns)
            superfluous_columns = set(actual_columns) - set(expected_columns)
            if columns_match := not missing_columns and not superfluous_columns:
                msg = "Table columns validated successfully."
            else:
                msg = "Table columns do not match reference columns!"
                if missing_columns:
                    msg += f"\n\tMissing columns: {sorted(missing_columns)!r}"
                if superfluous_columns:
                    msg += f"\n\tSuperfluous columns: {sorted(superfluous_columns)!r}"
    error_handler.emit(msg, valid=columns_match)

    # Validate dtypes (for matching columns only).
    match actual_dtypes, expected_dtypes:
        case None, None:
            msg_dtypes = "No dtypes in actual table and no reference dtypes given, skipping dtype validation."
            dtypes_match = True
        case _, None:
            msg_dtypes = "No reference dtypes given, skipping dtype validation."
            dtypes_match = False
        case None, _:
            msg_dtypes = "No dtypes in actual table, but reference dtypes given!"
            dtypes_match = False
        case _, _:
            assert actual_dtypes is not None  # needed for type checker
            assert expected_dtypes is not None  # needed for type checker
            bad_dtypes = {}
            for actual_col, actual_dtype in actual_dtypes.items():
                if actual_col not in expected_dtypes:
                    continue  # skip superfluous columns
                expected_dtype = expected_dtypes[actual_col]

                match actual_dtype, expected_dtype:
                    case None, None:
                        continue  # skip dtype validation for this column
                    case _, None:
                        continue  # skip dtype validation for this column
                    case None, _:
                        continue  # skip dtype validation for this column
                    case _, _:
                        if actual_dtype != expected_dtype:
                            bad_dtypes[actual_col] = (actual_dtype, expected_dtype)

            if dtypes_match := not bad_dtypes:
                msg_dtypes = "Table dtypes validated successfully."
            else:
                msg_dtypes = "Table dtypes do not match reference dtypes for columns:\n"
                for col, (actual_dtype, expected_dtype) in bad_dtypes.items():
                    msg_dtypes += (
                        f"\tColumn {col!r}:"
                        f" actual dtype={actual_dtype!r},"
                        f" expected dtype={expected_dtype!r}\n"
                    )

    error_handler.emit(msg_dtypes, valid=dtypes_match)

    return columns_match and dtypes_match
