r"""Hash function utils."""

__all__ = [
    # Constants
    "DEFAULT_HASH_METHOD",
    # Functions
    "hash_array",
    "hash_file",
    "hash_iterable",
    "hash_mapping",
    "hash_numpy",
    "hash_object",
    "hash_pandas",
    "hash_pyarrow",
    "hash_set",
    "to_alphanumeric",
    "to_base",
    "validate_file_hash",
    "validate_object_hash",
    "validate_table_hash",
    "validate_table_schema",
]

import hashlib
import logging
import string
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from types import NotImplementedType
from typing import Any, Final, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex, Series

from tsdm.constants import EMPTY_MAP
from tsdm.types.aliases import FilePath
from tsdm.types.protocols import SupportsShape

__logger__: logging.Logger = logging.getLogger(__name__)

DEFAULT_HASH_METHOD: Final[str] = "sha256"
r"""The default hash method to use."""


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


def to_alphanumeric(n: int) -> str:
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
    r"""Hash an object in a permutation invariant manner."""
    match x:
        case Hashable():
            return hash(x)
        case Index() | MultiIndex():
            return hash_pandas(x.to_frame())
        case DataFrame() | Series():
            return hash_pandas(x)
        case Mapping():
            return hash_mapping(x)
        case Iterable():
            return hash_iterable(x)
        case _:
            raise TypeError(f"Cannot hash object of type {type(x)}.")


def hash_set(x: Iterable[Hashable], /, *, ignore_duplicates: bool = False) -> int:
    r"""Hash an iterable of hashable objects in a permutation invariant manner."""
    if ignore_duplicates:
        return hash(frozenset(hash_object(y) for y in x))
    # We use counter as a proxy for multisets, do deal with duplicates in x.
    mdict = Counter(hash_object(y) for y in x)
    return hash(frozenset(mdict.items()))


def hash_mapping(x: Mapping[Hashable, Hashable], /) -> int:
    r"""Hash a Mapping of hashable objects in a permutation invariant manner."""
    return hash_set(x.items())


def hash_iterable(x: Iterable[Hashable], /) -> int:
    r"""Hash an iterable of hashable objects in an order dependent manner."""
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
    *,
    block_size: int = 65536,
    hash_algorithm: str = "sha256",
    **kwargs: Any,
) -> str:
    r"""Calculate the SHA256-hash of a file."""
    algorithms = vars(hashlib)
    hash_value = algorithms[hash_algorithm](**kwargs)

    with open(filepath, "rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(block_size), b""):
            hash_value.update(byte_block)

    return hash_value.hexdigest()


def hash_array(
    array: Any, *, hash_algorithm: str = "pandas", **hash_kwargs: Any
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


def validate_file_hash(
    file: FilePath | Sequence[FilePath] | Mapping[str, FilePath],
    reference: None | str | Mapping[str, str | None] = None,
    /,
    *,
    logger: logging.Logger = __logger__,
    errors: Literal["warn", "raise", "ignore"] = "warn",
    hash_alg: Optional[str] = None,
    hash_kwargs: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    r"""Validate file(s), given reference hash value(s).

    Args:
        file: The file to validate. If a mapping is given, every file in the mapping is validated.
        reference: The reference hash value, or a mapping from file names to
            (hash_algorithm, hash_value) pairs.
        hash_alg: The hash algorithm to use, if no reference hash is given.
        logger: Optional logger to use.
        errors: How to handle errors. Can be "warn", "raise", or "ignore".
        hash_kwargs: Additional keyword arguments to pass to the hash function.

    Raises:
        ValueError: If the file hash does not match the reference hash.
        LookupError: If the file is not found in the reference hash table.
    """
    if errors not in {"warn", "raise", "ignore"}:
        raise ValueError(
            f"Invalid value for errors: {errors!r}. "
            "Only 'warn', 'raise', and 'ignore' are allowed."
        )

    # If file is a sequence or mapping, recurse.
    match file:
        case str() | Path():  # must come before Iterable check!
            pass
        case Mapping():
            match reference:
                case Mapping() as mapping:
                    ref = mapping
                case None:
                    ref = {}
                case _:
                    raise TypeError(
                        f"Expected mapping or None, got {type(reference)} instead!"
                    )
            for key, value in file.items():
                validate_file_hash(
                    value,
                    ref.get(key, None),
                    hash_alg=hash_alg,
                    logger=logger,
                    errors=errors,
                    hash_kwargs=hash_kwargs,
                )
            return
        case Iterable():
            for f in file:
                validate_file_hash(
                    f,
                    reference,
                    hash_alg=hash_alg,
                    logger=logger,
                    errors=errors,
                    hash_kwargs=hash_kwargs,
                )
            return

    # Now, we know that file is a single file.
    if not isinstance(file, str | Path):
        raise TypeError(f"Expected str or Path, got {type(file)} instead!")
    file = Path(file)

    if isinstance(reference, Mapping):
        reference = reference.get(
            str(file), reference.get(file.name, reference.get(file.stem, None))
        )  # try to match by full path, then by name, then by stem

    # Determine the reference hash value.
    match reference:
        case None | NotImplementedType():
            ref_alg, ref_hash = None, None
        case str(name):
            # split hashes of the form "sha256:hash_value"
            # if no hash algorithm is given, check hash_algorithm
            ref_alg, ref_hash = name.rsplit(":", 1) if ":" in name else (None, name)
        case _:
            raise ValueError(f"Invalid reference {reference=} hash!")

    # Determine the hash algorithm to use.
    match ref_alg, hash_alg:
        case None, None:
            warnings.warn(
                "No hash algorithm given for reference hash!"
                f"Using {DEFAULT_HASH_METHOD!r} as default.",
                UserWarning,
                stacklevel=2,
            )
            hash_alg = DEFAULT_HASH_METHOD
        case str(name), _:
            hash_alg = name
        case None, str(name):
            hash_alg = name
        case _:
            raise ValueError("Invalid hash algorithm given!")

    # Compute the hash
    hash_value = hash_file(file, hash_algorithm=hash_alg, **hash_kwargs)

    # Finally, compare the hashes.
    if file.suffix == ".parquet":
        warnings.warn(
            f"{file!s} ✘✘ refusing to validate, since parquet is not binary stable!"
            f" Hash {hash_alg!s}:{hash_value!s}",
            UserWarning,
            stacklevel=2,
        )
    elif ref_hash is None:
        msg = (
            f"{file!s} ?? cannot be validated, since no reference hash is given!"
            f" Hash {hash_alg!s}:{hash_value!s}"
        )
        match errors:
            case "raise":
                raise LookupError(msg)
            case "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            case "ignore":
                logger.info(msg)
    elif hash_value != ref_hash:
        msg = (
            f"{file!s} ✘✘ failed the validation!"
            f" Hash {hash_alg!s}:{hash_value!s}"
            f" does not match reference {ref_hash!r}."
        )
        match errors:
            case "raise":
                raise LookupError(msg)
            case "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            case "ignore":
                logger.info(msg)
    else:
        logger.info(
            "%s ✔✔ successful validation! Hash %s:%s matches reference.",
            *(file, hash_alg, hash_value),
        )


def validate_table_hash(
    table: Any,
    reference: str | None = None,
    /,
    *,
    hash_algorithm: Optional[str] = None,
    logger: logging.Logger = __logger__,
    **hash_kwargs: Any,
) -> None:
    r"""Validate the hash of a `pandas` object, given hash values from a table."""
    # Try to determine the hash algorithm from the array type
    name = f"{type(table)} of shape={table.shape}"

    # Determine the reference hash
    match reference:
        case None:
            ref_alg, ref_hash = None, None
        case str(name):
            # split hashes of the form "sha256:hash_value"
            # if no hash algorithm is given, check hash_algorithm
            ref_alg, ref_hash = name.rsplit(":", 1) if ":" in name else (None, name)
        case _:
            raise ValueError(f"Invalid reference {reference=} hash!")

    # Determine the hash algorithm
    match ref_alg, hash_algorithm:
        case None, None:
            # Try to determine the hash algorithm from the array type
            match table:
                case DataFrame() | Series() | Index():
                    hash_algorithm = "pandas"
                case np.ndarray():
                    hash_algorithm = "numpy"
                case _:
                    hash_algorithm = DEFAULT_HASH_METHOD

            warnings.warn(
                "No hash algorithm given for reference hash!"
                f"Trying with {hash_algorithm!r} (auto-selected).",
                UserWarning,
                stacklevel=2,
            )
        case str(alg), _:
            hash_algorithm = alg
        case None, str(alg):
            hash_algorithm = alg
        case _:
            raise ValueError("Invalid hash algorithm given!")

    # Compute the hash.
    hash_value = hash_array(table, hash_algorithm=hash_algorithm, **hash_kwargs)

    # Finally, compare the hashes.
    if ref_hash is None:
        warnings.warn(
            f"No reference hash given for array-like object {name!r}."
            f"The {hash_algorithm!r}-hash is {hash_value!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif hash_value != ref_hash:
        warnings.warn(
            f"Array-Like object {name!r} failed to validate! {ref_alg!r}-hash-"
            f"value {hash_value!r} does not match reference {reference!r}."
            "Ignore this warning if the format is parquet.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        logger.info(
            "Array-Like object '%s' validated successfully with '%s'-hash '%s'.",
            name,
            hash_algorithm,
            hash_value,
        )


def validate_object_hash(
    obj: Any,
    reference: Any = None,
    /,
    *,
    hash_algorithm: Optional[str] = None,
    logger: logging.Logger = __logger__,
    **hash_kwargs: Any,
) -> None:
    r"""Validate object(s) given reference hash values."""
    # Try to determine the hash algorithm from the array type
    match obj:
        case str() | Path():
            validate_file_hash(
                obj,
                reference,
                hash_alg=hash_algorithm,
                logger=logger,
                **hash_kwargs,
            )
        case SupportsShape():
            validate_table_hash(
                obj,
                reference,
                hash_algorithm=hash_algorithm,
                logger=logger,
                **hash_kwargs,
            )
        case Mapping():  # apply recursively to all values
            match reference:
                case Mapping() as mapping:
                    ref = mapping
                case None:
                    ref = {}
                case _:
                    raise TypeError(
                        f"Expected mapping or None, got {type(reference)} instead!"
                    )
            for key, value in obj.items():
                validate_object_hash(
                    value,
                    ref.get(key, None),
                    hash_algorithm=hash_algorithm,
                    logger=logger,
                    **hash_kwargs,
                )
        case Iterable():
            match reference:
                case None:
                    for value in obj:  # recursion
                        validate_object_hash(
                            value,
                            None,
                            hash_algorithm=hash_algorithm,
                            logger=logger,
                            **hash_kwargs,
                        )
                case Iterable() as reference:
                    for value, ref in zip(obj, reference, strict=True):  # recursion
                        validate_object_hash(
                            value,
                            ref,
                            hash_algorithm=hash_algorithm,
                            logger=logger,
                            **hash_kwargs,
                        )
                case str():
                    raise TypeError("Cannot validate multiple items with single hash!")
                case _:
                    raise TypeError(f"Invalid reference hash type! {reference=}")
        case _:
            # Determine the reference hash value.
            match reference:
                case None:
                    reference_alg, reference_hash = None, None
                case str():
                    # split hashes of the form "sha256:hash_value"
                    # if no hash algorithm is given, check hash_algorithm
                    reference_alg, reference_hash = (
                        reference.rsplit(":", 1)
                        if ":" in reference
                        else (None, reference)
                    )
                case _:
                    raise ValueError(f"Invalid reference {reference=} hash!")

            # Compute the hash value of the object.
            hash_value = hash_object(obj)
            name = f"{type(obj)} {id(obj)}"
            if reference is None:
                warnings.warn(
                    f"No reference hash given for object {name!r}."
                    f"The {hash_algorithm!r}-hash is {hash_value!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif hash_value != reference_hash:
                warnings.warn(
                    f"Object {name!r} failed to validate! {reference_alg!r}-hash-"
                    f"value {hash_value!r} does not match reference {reference_hash!r}."
                    "Ignore this warning if the format is parquet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                logger.info(
                    "Object '%s' validated successfully with '%s'-hash '%s'.",
                    name,
                    hash_algorithm,
                    hash_value,
                )


def validate_table_schema(
    table: SupportsShape,
    /,
    *,
    reference_shape: Optional[tuple[int, int]] = None,
    reference_schema: Optional[Sequence[str] | Mapping[str, str] | pa.Schema] = None,
) -> None:
    r"""Validate the schema of a `pandas` object, given schema values from a table.

    Check if the columns and dtypes of the table match the reference schema.
    Check if the shape of the table matches the reference schema.
    """
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
    if reference_shape is None:
        __logger__.info("No reference shape given, skipping shape validation.")
    elif shape != reference_shape:
        raise ValueError(f"Table {shape=!r} does not match {reference_shape=!r}!")
    else:
        __logger__.info("Table shape validated successfully.")

    # Validate columns.
    if reference_columns is None:
        __logger__.info("No reference columns given, skipping column validation.")
    elif columns != reference_columns:
        raise ValueError(f"Table {columns=!r} does not match {reference_columns=!r}!")
    else:
        __logger__.info("Table columns validated successfully.")

    # Validate dtypes.
    if reference_dtypes is None:
        __logger__.info("No reference dtypes given, skipping dtype validation.")
    elif dtypes != reference_dtypes:
        raise ValueError(f"Table {dtypes=!r} does not match {reference_dtypes=!r}!")
    else:
        __logger__.info("Table dtypes validated successfully.")
