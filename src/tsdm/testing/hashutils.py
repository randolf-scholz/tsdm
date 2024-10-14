r"""Hash function utils."""

__all__ = [
    # Constants
    "DEFAULT_HASH_METHOD",
    "HASH_REGEX",
    # Classes
    "ValidationError",
    "Hash",
    "ErrorHandler",
    # Functions
    "hash_array",
    "hash_file",
    "hash_collection",
    "hash_mapping",
    "hash_numpy",
    "hash_object",
    "hash_pandas",
    "hash_pyarrow",
    "hash_set",
    "to_alphanumeric",
    "to_base",
    "to_int",
    "validate_hash",
    "validate_file",
    "validate_table",
    "validate_table_schema",
]

import hashlib
import logging
import re
import string
import warnings
from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, Optional, Self

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex, Series

from tsdm.constants import EMPTY_MAP
from tsdm.types.aliases import FilePath
from tsdm.types.arrays import SupportsShape

__logger__: logging.Logger = logging.getLogger(__name__)

DEFAULT_HASH_METHOD: Final[str] = "sha256"
r"""The default hash method to use."""

HASH_REGEX: Final[re.Pattern] = re.compile(r"^(?:(?P<alg>\w+):)?(?P<value>[a-f0-9]+)$")
r"""Regular expression to match hash values."""


class Hash(NamedTuple):
    r"""Hash value with optional algorithm."""

    hash_value: str
    hash_algorithm: str | None

    @classmethod
    def from_value(cls, arg: str | Self, /) -> Self:
        if isinstance(arg, cls):
            return arg

        s: str = arg  # type: ignore[assignment]
        s = s.lower()
        # match against the hash regex
        match = HASH_REGEX.match(s)

        # ensure that there is precisely one match
        if match is None or match.pos != 0 or match.endpos != len(s):
            raise ValueError(f"Invalid hash value: {s!r}")

        return cls(hash_value=match["value"], hash_algorithm=match["alg"])

    def to_int(self, *, base: Optional[int] = None) -> int:
        return to_int(self.hash_value, base=base)

    def __str__(self) -> str:
        if self.hash_algorithm is None:
            return self.hash_value
        return f"{self.hash_algorithm}:{self.hash_value}"


class ValidationError(ValueError):
    r"""Validation error for hash validation."""


class ErrorHandler:
    r"""Validation mode for hash validation."""

    class MODE(StrEnum):
        r"""Validation mode for hash validation."""

        IGNORE = "ignore"
        LOG = "log"
        WARN = "warn"
        RAISE = "raise"

    type Mode = Literal["ignore", "log", "warn", "raise"] | MODE

    mode: MODE
    r"""Validation mode for hash validation."""

    def __init__(self, mode: MODE | Literal["ignore", "log", "warn", "raise"]) -> None:
        self.mode = self.MODE(mode)

    def emit(self, msg: str, /, *, valid: bool = False) -> None:
        match self.mode, valid:
            case "raise", False:
                raise ValidationError(msg)
            case "warn", False:
                warnings.warn(msg, UserWarning, stacklevel=2)
            case "log", _:
                __logger__.info(msg)
            case "ignore":
                pass


def to_int(value: str, /, *, base: Optional[int] = None) -> int:
    r"""Convert a string to an integer, autodetecting the base if necessary."""
    # if only decimals:
    if base is None:
        if all(c in "01" for c in value):  # binary
            base = 2
        elif all(c in "01234567" for c in value):  # octal
            base = 8
        elif all(c.isdigit() for c in value):  # decimal
            base = 10
        elif all(c in string.hexdigits for c in value):  # hexadecimal
            base = 16
        elif all(
            c in string.ascii_lowercase + string.digits for c in value
        ):  # alphanumeric
            base = 36
        else:
            raise ValueError(f"Could not autodetect base for {value!r}.")
    return int(value, base=base)


def to_base(n: int, base: int, /) -> list[int]:
    r"""Convert non-negative integer to any basis.

    The result satisfies: ``n = sum(d*b**k for k, d in enumerate(reversed(digits)))``

    References:
        - https://stackoverflow.com/a/28666223
    """
    if n < 0:
        raise ValueError("n must be non-negative!")

    digits = []
    while n:
        n, d = divmod(n, base)
        digits.append(d)
    return digits[::-1] or [0]


def to_alphanumeric(n: int, /) -> str:
    r"""Convert integer to alphanumeric code.

    Note:
        We assume n is an int64. We first convert it to uint64, then to base 36.
        Doubling the alphabet size generally reduces the length of the code by
        a factor of $1 + 1/log₂B$, where $B$ is the alphabet size.
    """
    # int64  range: [-2⁶³, 2⁶³-1] = [-9,223,372,036,854,775,808, +9,223,372,036,854,775,807]
    # uint64 range: [0,    2⁶⁴-1] = [0, 18446744073709551615]
    chars = string.ascii_uppercase + string.digits
    digits = to_base(n + 2**63, len(chars))
    return "".join(chars[i] for i in digits)


def hash_object(x: Any, /) -> int:
    r"""Hash an object."""
    match x:
        case Hashable():
            return hash(x)
        case Index() | MultiIndex():
            return hash_pandas(x.to_frame())
        case DataFrame() | Series():
            return hash_pandas(x)
        case Mapping():
            return hash_mapping(x)
        case Collection():
            return hash_collection(x)
        case _:
            raise TypeError(f"Cannot hash object of type {type(x)}.")


def hash_set(x: Iterable[Hashable], /, *, ignore_duplicates: bool = False) -> int:
    r"""Hash an `Iterable` of hashable objects in a permutation invariant manner."""
    if ignore_duplicates:
        return hash(frozenset(hash_object(y) for y in x))
    # We use counter as a proxy for multisets, do deal with duplicates in x.
    mdict = Counter(hash_object(y) for y in x)
    return hash(frozenset(mdict.items()))


def hash_mapping(x: Mapping[Hashable, Hashable], /) -> int:
    r"""Hash a `Mapping` of hashable objects in a permutation invariant manner."""
    return hash_set(x.items())


def hash_collection(x: Collection[Hashable], /) -> int:
    r"""Hash a `Collection` of hashable objects in an order-dependent manner."""
    return hash_set(enumerate(x))


def hash_pandas(
    df: Index | Series | DataFrame,
    /,
    *,
    index: bool = True,
    ignore_duplicates: bool = False,
    row_invariant: bool = False,
    col_invariant: bool = False,
) -> int:
    r"""Hash pandas object to a single number.

    If row_invariant is True, then the hash is invariant to the order of the rows.
    If col_invariant is True, then the hash is invariant to the order of the columns.

    .. math:: hash(PX) = hash(X) and hash(XQ) = hash(X)

    for permutation matrices $P$ and $Q$ respectively.
    """
    # TODO: problem: duplicate rows ⇝ include standard index for rows/cols!

    if col_invariant:
        df = df.stack()

    row_hash = pd.util.hash_pandas_object(df, index=index)

    hash_value = (
        hash_set(row_hash, ignore_duplicates=ignore_duplicates)
        if row_invariant
        else hash(tuple(row_hash))
    )

    return hash_value


def hash_numpy(array: NDArray, /) -> int:
    r"""Hash a numpy array."""
    # TODO: there are probably better ways to hash numpy arrays.
    array = np.asarray(array)
    hash_value = hash(tuple(array.flatten()))
    return hash_value


def hash_pyarrow(array: pa.Array, /) -> int:
    r"""Hash a pyarrow array."""
    raise NotImplementedError(f"Can't hash {type(array)} yet.")


def hash_file(
    filepath: FilePath,
    /,
    *,
    block_size: int = 65536,
    hash_algorithm: str = "sha256",
    **hash_kwargs: Any,
) -> Hash:
    r"""Calculate the SHA256-hash of a file."""
    algorithms = vars(hashlib)
    hasher = algorithms[hash_algorithm](**hash_kwargs)

    with open(filepath, "rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(block_size), b""):
            hasher.update(byte_block)

    hash_value = hasher.hexdigest()
    return Hash(hash_value=hash_value, hash_algorithm=hash_algorithm)


def hash_array(
    array: Any, /, *, hash_algorithm: str = "pandas", **hash_kwargs: Any
) -> str:
    r"""Hash an array like object (pandas/numpy/pyarrow/etc.)."""
    match hash_algorithm:
        case "pandas":
            hash_value = hash_pandas(array, **hash_kwargs)
        case "numpy":
            hash_value = hash_numpy(array, **hash_kwargs)
        case "pyarrow":
            hash_value = hash_pyarrow(array, **hash_kwargs)
        case _:
            raise NotImplementedError(f"No algorithm {hash_algorithm!r} known.")

    formatter = to_alphanumeric
    return formatter(hash_value)


def validate_hash(
    hash_value: Hash | str,
    reference: Hash | str | None = None,
    /,
    *,
    prefix: str = "",
    errors: ErrorHandler.Mode = "raise",
) -> bool:
    r"""Compare a hash value against a reference hash value."""
    error_handler = ErrorHandler(errors)
    left = Hash.from_value(hash_value)
    right = Hash.from_value(reference) if reference is not None else None
    valid_hash = left == right

    # Determine the message to emit.
    if right is None:
        msg = f"✘ No reference hash given for {left!s}."
    elif right.hash_algorithm != left.hash_algorithm:
        msg = f"✘ Hash algorithm mismatch: {left.hash_algorithm!r} ≠ {right.hash_algorithm!r}."
    elif right.hash_value != left.hash_value:
        msg = f"✘ Hash mismatch: {left!s} ≠ {right!s}."
    else:
        msg = f"✔ Hash {left!s} matches reference {right!s}."

    error_handler.emit(prefix + msg, valid=valid_hash)
    return valid_hash


def validate_file(
    filepath: FilePath,
    reference: Hash | str | None = None,
    /,
    *,
    hash_algorithm: Optional[str] = None,
    hash_kwargs: Mapping[str, Any] = EMPTY_MAP,
    errors: ErrorHandler.Mode = "raise",
) -> bool:
    r"""Validate file(s), given reference hash value(s).

    Args:
        filepath: The file to validate.
        reference: The reference hash value (optional).
        hash_algorithm: The hash algorithm to use.
        hash_kwargs: Additional keyword arguments to pass to the hash function.
        errors: How to handle errors. Can be "warn", "raise", "log" or "ignore". (default: "warn")

    Raises:
        ValueError: If the file hash does not match the reference hash.
        LookupError: If the file is not found in the reference hash table.
    """
    # region input validation ----------------------------------------------------------
    error_handler = ErrorHandler(errors)
    file = Path(filepath)
    valid_hash: bool = False

    if file.suffix == ".parquet":
        error_handler.emit(
            f"{file!s}: ✘✘ refusing to hash, since parquet is not binary stable!",
            valid=valid_hash,
        )
        return valid_hash

    # Determine the reference hash value.
    right = Hash.from_value(reference) if reference is not None else None

    # Determine the hash algorithm to use.
    match hash_algorithm, right:
        case None, None | Hash(hash_algorithm=None):
            warnings.warn(
                "No hash algorithm given for reference hash!"
                f"Using {DEFAULT_HASH_METHOD!r} as default.",
                UserWarning,
                stacklevel=2,
            )
            hash_alg = DEFAULT_HASH_METHOD
        case None, Hash(hash_algorithm=str(ref_alg)):
            hash_alg = ref_alg
        case str(alg), _:
            hash_alg = alg
        case _:
            raise TypeError(f"Invalid hash algorithm type: {type(hash_algorithm)}")

    # Compute the hash
    left = hash_file(file, hash_algorithm=hash_alg, **hash_kwargs)
    return validate_hash(left, right, errors=errors, prefix=f"{file!s}: ")


def validate_table(
    table: Any,
    reference: Hash | str | None = None,
    /,
    *,
    errors: ErrorHandler.Mode = "warn",
    hash_algorithm: Optional[str] = None,
    **hash_kwargs: Any,
) -> bool:
    r"""Validate the hash of a `pandas` object, given hash values from a table."""
    # Try to determine the hash algorithm from the array type
    cls = type(table)
    name = f"{cls} of shape={table.shape}"
    right = Hash.from_value(reference) if reference is not None else None

    # Determine the hash algorithm
    match hash_algorithm, right:
        case None, None | Hash(hash_algorithm=None):
            # Try to determine the hash algorithm from the array type
            warnings.warn(
                "No hash algorithm given for reference hash! Trying to auto-select.",
                UserWarning,
                stacklevel=2,
            )

            match table:
                case DataFrame() | Series() | Index():
                    hash_alg = "pandas"
                case np.ndarray():
                    hash_alg = "numpy"
                case pa.Array():
                    hash_alg = "pyarrow"
                case _:
                    raise TypeError(
                        f"Could not autodetect hash algorithm for table of type {cls}"
                    )

        case None, Hash(hash_algorithm=str(ref_alg)):
            hash_alg = ref_alg
        case str(alg), _:
            hash_alg = alg
        case _:
            raise TypeError(f"Invalid hash algorithm type: {type(hash_algorithm)}")

    # Compute the hash.
    left = hash_array(table, hash_algorithm=hash_alg, **hash_kwargs)
    return validate_hash(left, right, errors=errors, prefix=f"{name!s}: ")


def validate_table_schema(
    table: SupportsShape,
    /,
    *,
    reference_shape: Optional[tuple[int, int]] = None,
    reference_schema: Optional[Sequence[str] | Mapping[str, str] | pa.Schema] = None,
    errors: ErrorHandler.Mode = "warn",
) -> bool:
    r"""Validate the schema of a `pandas` object, given schema values from a table.

    Check if the columns and dtypes of the table match the reference schema.
    Check if the shape of the table matches the reference schema.
    """
    error_handler = ErrorHandler(errors)

    # get data shape, columns and dtypes from table
    match table:
        case MultiIndex() as multiindex:
            shape = multiindex.shape
            columns = multiindex.names
            dtypes = multiindex.dtypes
        case Index() as index:
            shape = index.shape
            columns = index.name
            dtypes = index.dtype
        case Series() as series:
            shape = series.shape
            columns = series.name
            dtypes = series.dtype
        case DataFrame() as df:
            shape = df.shape
            columns = df.columns.tolist()
            dtypes = df.dtypes.tolist()
        case pa.Table() as arrow_table:
            shape = arrow_table.shape
            columns = arrow_table.column_names
            dtypes = arrow_table.schema.types
        case np.ndarray() as arr:
            shape = arr.shape
            columns = None
            dtypes = [arr.dtype]
        case _:
            raise NotImplementedError(
                f"Cannot validate schema for {type(table)} objects!"
            )

    # get reference columns and dtypes
    match reference_schema:
        case pa.Schema() as schema:
            reference_columns = schema.names
            reference_dtypes = schema.types
        case Mapping() as mapping:
            reference_columns = list(mapping.keys())
            reference_dtypes = list(mapping.values())
        case Sequence() as seq:
            reference_columns = list(seq)
            reference_dtypes = None
        case None:
            reference_columns = None
            reference_dtypes = None
        case _:
            raise TypeError(f"Invalid reference schema type! {type(reference_schema)=}")

    # Validate shape.
    valid_shape = shape == reference_shape
    msg_shape = (
        "No reference shape given, skipping shape validation."
        if reference_shape is None
        else f"Table {shape=!r} does not match {reference_shape=!r}!"
        if not valid_shape
        else "Table shape validated successfully."
    )
    error_handler.emit(msg_shape, valid=valid_shape)

    # Validate columns.
    valid_columns = columns == reference_columns
    msg_columns = (
        "No reference columns given, skipping column validation."
        if reference_columns is None
        else f"Table {columns=!r} does not match {reference_columns=!r}!"
        if not valid_columns
        else "Table columns validated successfully."
    )
    error_handler.emit(msg_columns, valid=valid_columns)

    # Validate dtypes.
    valid_dtypes = dtypes == reference_dtypes
    msg_dtypes = (
        "No reference dtypes given, skipping dtype validation."
        if reference_dtypes is None
        else f"Table {dtypes=!r} does not match {reference_dtypes=!r}!"
        if not valid_dtypes
        else "Table dtypes validated successfully."
    )
    error_handler.emit(msg_dtypes, valid=valid_dtypes)

    return valid_shape and valid_columns and valid_dtypes
