r"""Validation utilities for hash and schema validation."""

__all__ = [
    # classes
    "ErrorHandler",
    "ValidationError",
    # functions
    "make_error_handler",
    "validate_csv_header",
    "validate_file_hash",
    "validate_hash",
    "validate_table_schema",
    "validate_table_shape",
]

import csv
import hashlib
import logging
import warnings
from collections.abc import Iterable, Mapping
from enum import StrEnum
from io import TextIOWrapper
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Literal, assert_never, overload

from . import config
from .hashing import Hash, hash_file
from .types import FilePath, FileStream, HasShape


class ValidationError(ValueError):
    r"""Exception raised when a hash, shape, or schema check fails."""


class ErrorHandler:
    r"""Handle validation messages by ignoring, logging, warning, or raising."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")

    class MODE(StrEnum):
        r"""Supported ways to handle validation messages."""

        IGNORE = "ignore"
        LOG = "log"
        WARN = "warn"
        RAISE = "raise"

    # type alias for handling both enums and string literals.
    # SEE: https://discuss.python.org/t/amend-pep-586-to-make-enum-values-subtypes-of-literal/59456
    type Mode = "MODE" | Literal["ignore", "log", "warn", "raise"]

    mode: MODE
    r"""Selected validation message policy."""

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
        r"""Emit a validation message according to the selected mode.

        Raise or warn only when `valid=False`. Log mode records both successful
        and failed checks, while ignore mode discards all messages.
        """
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
    r"""Return an existing error handler or create one from a mode.

    When `arg` is already an `ErrorHandler`, return it unchanged and ignore
    `prefix` and `postfix`.
    """
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

    Compare digest strings exactly, including case. Algorithm-name differences
    are reported in the diagnostic message; the result depends on the digest
    strings. Two missing hashes pass, while exactly one missing hash fails.

    Args:
        hash_value: The hash value to validate.
        expected_hash: The reference hash value.
        errors: Existing error handler or one of `"ignore"`, `"log"`, `"warn"`,
            and `"raise"`.

    Returns:
        Whether the digest strings match, or both hashes are missing.

    Raises:
        ValidationError: If `errors="raise"` and the digest comparison fails.
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
    hash_algorithm: str | None = None,
    errors: ErrorHandler.Mode = "raise",
    skipif_no_reference: bool = False,
) -> bool:
    r"""Validate a file or binary stream against a reference hash.

    Paths ending in `.parquet` are rejected because their binary representation
    is not stable. Streams must support seeking; their original position is
    restored even if reading fails.

    Args:
        path_or_stream: The file or binary stream to validate. Streams are hashed from the
            beginning and returned to their original position afterwards.
        expected_hash: The reference hash value.
        hash_algorithm: Algorithm to use. If omitted, use the reference algorithm
            when available; otherwise warn and use `config.DEFAULT_HASH_METHOD`.
        errors: Message policy: `"warn"`, `"raise"`, `"log"`, or `"ignore"`.
            Defaults to `"raise"`.
        skipif_no_reference: Whether to return `True` without hashing when the
            reference hash is missing.

    Returns:
        Whether the digest matches, or validation was explicitly skipped.

    Raises:
        ValidationError: If `errors="raise"` and the file is rejected or its
            digest does not match the reference.
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
                        f"Using {config.DEFAULT_HASH_METHOD!r} as default.",
                        UserWarning,
                        stacklevel=2,
                    )
                    hash_alg = config.DEFAULT_HASH_METHOD
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


def validate_table_shape(
    table: HasShape,
    /,
    expected_shape: tuple[int, ...] | None,
    *,
    errors: ErrorHandler.Mode = "warn",
) -> bool:
    r"""Compare an object's shape with a reference shape.

    A missing reference shape skips comparison and returns `True`. Otherwise,
    return whether the shapes match, handling failures according to `errors`.

    Raises:
        TypeError: If the object does not expose a supported shape.
        ValidationError: If `errors="raise"` and the shapes differ.
    """
    error_handler = make_error_handler(errors)

    match table:
        case HasShape(shape=actual_shape):
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


def _infer_schema(table: Any, /) -> Mapping[Any, Any]:
    r"""Infer a mapping from column names to dtypes for a table-like object.

    A table's index is not included. Pass an index directly to infer its schema.

    Returns:
        A mapping from column names to dtypes.

    Raises:
        NotImplementedError: If the table type is not supported.
    """
    match table:
        case object(schema=Mapping() as schema):  # type: ignore
            # pl.DataFrame
            return dict(schema)

        case object(dtypes=Iterable() as dtypes):  # type: ignore
            # pd.MultiIndex, pd.DataFrame
            return dict(dtypes)

        case object(name=name, dtype=dtype):  # type: ignore
            # pd.Series, pd.Index, pl.Series
            return {name: dtype}

        case object(schema=schema):  # type: ignore
            # pa.Table
            return dict(zip(schema.names, schema.types, strict=True))

        case _:
            raise NotImplementedError(f"Cannot get schema for {type(table)} objects!")


def validate_table_schema(
    table: Any,
    /,
    *,
    expected_schema: Mapping[str, Any | None],
    errors: ErrorHandler.Mode = "warn",
) -> bool:
    r"""Compare a table's column names and dtypes with a reference schema.

    Column order is ignored. Column names must match exactly; index columns
    are not inspected separately. An expected dtype of `None` skips the dtype
    check for that column, but the column must still be present.

    Args:
        table: The table to validate.
        expected_schema: Mapping from column names to expected dtypes. Use
            `None` for a names-only check of an individual column.
        errors: Message policy: `"ignore"`, `"log"`, `"warn"`, or `"raise"`.

    Returns:
        Whether both column and dtype checks pass.

    Raises:
        ValidationError: If `errors="raise"` and either schema check fails.
    """
    error_handler = make_error_handler(errors)

    actual_schema = _infer_schema(table)

    # Validate columns.
    missing_columns = set(expected_schema) - set(actual_schema)
    superfluous_columns = set(actual_schema) - set(expected_schema)
    columns_match = not missing_columns and not superfluous_columns
    if columns_match:
        msg = "Table columns validated successfully."
    else:
        msg = "Table columns do not match reference columns!"
        if missing_columns:
            msg += f"\n\tMissing columns: {sorted(missing_columns)!r}"
        if superfluous_columns:
            msg += f"\n\tSuperfluous columns: {sorted(superfluous_columns)!r}"
    error_handler.emit(msg, valid=columns_match)

    # Validate dtypes (for matching columns only).
    bad_dtypes = {
        col: (actual_schema[col], expected_dtype)
        for col, expected_dtype in expected_schema.items()
        if col in actual_schema
        and expected_dtype is not None
        and actual_schema[col] != expected_dtype
    }
    dtypes_match = not bad_dtypes
    if dtypes_match:
        msg_dtypes = "Table dtypes validated successfully."
    else:
        msg_dtypes = "Table dtypes do not match reference dtypes for columns:\n"
        msg_dtypes += "".join(
            f"\tColumn {col!r}: actual dtype={actual_dtype!r},"
            f" expected dtype={expected_dtype!r}\n"
            for col, (actual_dtype, expected_dtype) in bad_dtypes.items()
        )

    error_handler.emit(msg_dtypes, valid=dtypes_match)

    return columns_match and dtypes_match


def validate_csv_header(
    file: FilePath | FileStream,
    schema: Iterable[str],
    /,
    *,
    separator: str = ",",
) -> None:
    r"""Validate that a CSV header exactly matches a schema.

    Read the first CSV record as UTF-8, allowing a byte-order mark. Streams
    are read from their current position and restored afterwards, even on
    failure, and remain open.

    Args:
        file: CSV path or seekable file-like object.
        schema: Expected column names, in CSV order.
        separator: CSV field separator.

    Raises:
        ValueError: If the file is not seekable or its header does not match the
            schema's columns and order.
    """
    match file:
        case str() | PathLike():
            with open(file, "rb") as stream:
                validate_csv_header(stream, schema, separator=separator)
            return
        case stream:
            if not stream.seekable():
                raise ValueError(
                    "Cannot validate the header of a non-seekable CSV stream."
                )

            position = stream.tell()
            try:
                text_stream = TextIOWrapper(stream, encoding="utf-8-sig", newline="")
                try:
                    reader = csv.reader(text_stream, delimiter=separator, strict=True)
                    actual_columns = next(reader, None)
                finally:
                    text_stream.detach()
            finally:
                stream.seek(position)

    if actual_columns is None:
        raise ValueError("CSV file is empty; expected a header.")

    expected_columns = list(schema)
    missing_cols = set(expected_columns) - set(actual_columns)
    superfluous_cols = set(actual_columns) - set(expected_columns)
    if missing_cols or superfluous_cols:
        raise ValueError(
            "CSV header does not match schema: "
            f"\n\t    missing: {sorted(missing_cols)!r}"
            f"\n\tsuperfluous: {sorted(superfluous_cols)!r}"
        )

    if len(actual_columns) != len(expected_columns):
        raise ValueError(
            "CSV header and schema have different numbers of columns: "
            f"\n\texpected: {expected_columns!r}"
            f"\n\t  actual: {actual_columns!r}"
        )

    if actual_columns != expected_columns:
        raise ValueError(
            "CSV header and schema columns are in different orders: "
            f"\n\texpected: {expected_columns!r}"
            f"\n\t  actual: {actual_columns!r}"
        )
