r"""Hash function utils."""

__all__ = [
    # Functions
    "hash_array",
    "hash_file",
    "hash_iterable",
    "hash_mapping",
    "hash_numpy",
    "hash_object",
    "hash_pandas",
    "hash_pandas",
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
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex, Series

from tsdm.types.aliases import PathLike, Schema
from tsdm.types.protocols import Table

__logger__ = logging.getLogger(__name__)

DEFAULT_HASH_METHOD = "sha256"
"""The default hash method to use."""


def to_base(n: int, b: int) -> list[int]:
    r"""Convert non-negative integer to any basis.

    The result satisfies: ``n = sum(d*b**k for k, d in enumerate(reversed(digits)))``

    References:
        - https://stackoverflow.com/a/28666223/9318372
    """
    digits = []
    while n:
        n, d = divmod(n, b)
        digits.append(d)
    return digits[::-1] or [0]


def to_alphanumeric(n: int) -> str:
    r"""Convert integer to alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    digits = to_base(n, len(chars))
    return "".join(chars[i] for i in digits)


def hash_object(x: Any, /) -> int:
    r"""Hash an object in a permutation invariant manner."""
    if isinstance(x, Hashable):
        return hash(x)
    if isinstance(x, Index | MultiIndex):
        return hash_pandas(x.to_frame())
    if isinstance(x, DataFrame | Series):
        return hash_pandas(x)
    if isinstance(x, Mapping):
        return hash_mapping(x)
    if isinstance(x, Iterable):
        return hash_iterable(x)
    raise TypeError(f"Cannot hash object of type {type(x)}.")


def hash_set(x: Iterable[Hashable], /, *, ignore_duplicates: bool = False) -> int:
    r"""Hash an iterable of hashable objects in a permutation invariant manner."""
    if ignore_duplicates:
        return hash(frozenset(hash_object(y) for y in x))
    # We use counter as a proxy for multisets, do deal with duplicates in x.
    mdict = Counter((hash_object(y) for y in x))
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

    if row_invariant:
        hash_value = hash_set(row_hash, ignore_duplicates=ignore_duplicates)
    else:
        hash_value = hash(tuple(row_hash))

    return hash_value


def hash_numpy(array: NDArray, /) -> int:
    """Hash a numpy array."""
    # TODO: there are probably better ways to hash numpy arrays.
    array = np.asarray(array)
    hash_value = hash(tuple(array.flatten()))
    return hash_value


def hash_file(
    file: PathLike,
    *,
    block_size: int = 65536,
    hash_algorithm: str = "sha256",
    **kwargs: Any,
) -> str:
    r"""Calculate the SHA256-hash of a file."""
    algorithms = vars(hashlib)
    hash_value = algorithms[hash_algorithm](**kwargs)

    with open(file, "rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(block_size), b""):
            hash_value.update(byte_block)

    return hash_value.hexdigest()


def hash_array(
    array: Any, *, hash_algorithm: str = "pandas", **hash_kwargs: Any
) -> int:
    """Hash an array like object (pandas/numpy/pyarrow/etc)."""
    if hash_algorithm == "pandas":
        return hash_pandas(array, **hash_kwargs)
    elif hash_algorithm == "numpy":
        return hash_numpy(array, **hash_kwargs)

    raise NotImplementedError(f"No algorithm {hash_algorithm!r} known.")


def validate_file_hash(
    file: PathLike | Sequence[PathLike] | Mapping[str, PathLike],
    /,
    reference: None | str | Mapping[str, str] = None,
    *,
    hash_algorithm: Optional[str] = None,
    logger: logging.Logger = __logger__,
    errors: Literal["warn", "raise", "ignore"] = "warn",
    **hash_kwargs: Any,
) -> None:
    """Validate file(s), given reference hash value(s).

    Arguments:
        file: The file to validate. If a mapping is given, every file in the mapping is validated.
        reference: The reference hash value, or a mapping from file names to
            (hash_algorithm, hash_value) pairs.
        hash_algorithm: The hash algorithm to use, if no reference hash is given.
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
            for key, value in file.items():
                if isinstance(reference, Mapping):
                    validate_file_hash(
                        value,
                        reference=reference.get(key, None),
                        hash_algorithm=hash_algorithm,
                        logger=logger,
                        errors=errors,
                        **hash_kwargs,
                    )
                elif reference is None:
                    validate_file_hash(
                        value,
                        reference=reference,
                        hash_algorithm=hash_algorithm,
                        logger=logger,
                        errors=errors,
                        **hash_kwargs,
                    )
                else:
                    raise ValueError(
                        "If a mapping of files is provided, reference_hash must be a mapping as well!"
                    )
            return
        case Iterable():
            for f in file:
                validate_file_hash(
                    f,
                    reference=reference,
                    hash_algorithm=hash_algorithm,
                    logger=logger,
                    errors=errors,
                    **hash_kwargs,
                )
            return

    # Now, we know that file is a single file.
    assert isinstance(file, str | Path)
    file = Path(file)

    if file.suffix == ".parquet":
        warnings.warn(
            f"Parquet file {file.name!r} not validated since the format is not binary stable!",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(reference, Mapping):
        reference = reference.get(
            str(file), reference.get(file.name, reference.get(file.stem, None))
        )  # try to match by full path, then by name, then by stem

    # Determine the reference hash value.
    match reference:
        case None:
            reference_alg, reference_hash = None, None
        case str():
            # split hashes of the form "sha256:hash_value"
            # if no hash algorithm is given, check hash_algorithm
            reference_alg, reference_hash = (
                reference.rsplit(":", 1) if ":" in reference else (None, reference)
            )
        case _:
            raise ValueError(f"Invalid reference {reference=} hash!")

    # Determine the hash algorithm to use.
    match reference_alg, hash_algorithm:
        case None, None:
            warnings.warn(
                "No hash algorithm given for reference hash!"
                f"Using {DEFAULT_HASH_METHOD!r} as default.",
                UserWarning,
                stacklevel=2,
            )
            hash_algorithm = DEFAULT_HASH_METHOD
        case str() as alg, _:
            hash_algorithm = alg
        case None, str() as alg:
            hash_algorithm = alg
        case _:
            raise ValueError("Invalid hash algorithm given!")

    # Compute the hash
    hash_value = hash_file(file, hash_algorithm=hash_algorithm, **hash_kwargs)

    # Finally, compare the hashes.
    if reference_hash is None:
        msg = (
            f"No reference hash given for file {file.name!r}."
            f"The {hash_algorithm!r}-hash is {hash_value!r}."
        )
        if errors == "raise":
            raise LookupError(msg)
        elif errors == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            logger.info(msg)
    elif hash_value != reference_hash:
        msg = (
            f"File {file.name!r} failed to validate! {reference_alg!r}-hash-"
            f"value {hash_value!r} does not match reference {reference_hash!r}."
            f"Ignore this warning if the format is parquet."
        )
        if errors == "raise":
            raise ValueError(msg)
        elif errors == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            logger.info(msg)
    else:
        logger.info(
            "File '%s' validated successfully with '%s'-hash '%s'.",
            *(file, hash_algorithm, hash_value),
        )


def validate_table_hash(
    table: Table,
    /,
    reference: str | None,
    *,
    hash_algorithm: Optional[str] = None,
    logger: logging.Logger = __logger__,
    **hash_kwargs: Any,
) -> None:
    """Validate the hash of a pandas object, given hash values from a table."""
    # Try to determine the hash algorithm from the array type
    name = f"{type(table)}<{table.shape}> {id(table)}"

    # Determine the reference hash
    match reference:
        case None:
            reference_alg, reference_hash = None, None
        case str():
            # split hashes of the form "sha256:hash_value"
            # if no hash algorithm is given, check hash_algorithm
            reference_alg, reference_hash = (
                reference.rsplit(":", 1) if ":" in reference else (None, reference)
            )
        case _:
            raise ValueError(f"Invalid reference {reference=} hash!")

    # Determine the hash algorithm
    match reference_alg, hash_algorithm:
        case None, None:
            # Try to determine the hash algorithm from the array type
            match table:
                case pd.DataFrame() | pd.Series() | pd.Index():
                    hash_algorithm = "pandas"
                case np.ndarray():
                    hash_algorithm = "numpy"
                case _:
                    raise ValueError(
                        f"Cannot determine hash algorithm for array-like object {table!r}."
                        "Please specify the hash algorithm manually."
                    )
            warnings.warn(
                "No hash algorithm given for reference hash!"
                f"Using {DEFAULT_HASH_METHOD!r} as default.",
                UserWarning,
                stacklevel=2,
            )
            hash_algorithm = DEFAULT_HASH_METHOD
        case str() as alg, _:
            hash_algorithm = alg
        case None, str() as alg:
            hash_algorithm = alg
        case _:
            raise ValueError("Invalid hash algorithm given!")

    # Compute the hash.
    hash_value = hash_array(table, hash_algorithm=hash_algorithm, **hash_kwargs)

    # Finally, compare the hashes.
    if reference_hash is None:
        warnings.warn(
            f"No reference hash given for array-like object {name!r}."
            f"The {hash_algorithm!r}-hash is {hash_value!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif hash_value != reference_hash:
        warnings.warn(
            f"Array-Like object {name!r} failed to validate! {reference_alg!r}-hash-"
            f"value {hash_value!r} does not match reference {reference!r}."
            f"Ignore this warning if the format is parquet.",
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


def validate_table_schema(table: Table, /, reference_schema: Schema) -> None:
    """Validate the schema of a pandas object, given schema values from a table.

    Checks if the columns and dtypes of the table match the reference schema.
    Checks if the shape of the table matches the reference schema.
    """
    # get shape, columns and dtypes from table
    match table:
        case pd.MultiIndex() as multiindex:
            shape = multiindex.shape
            columns = multiindex.names
            dtypes = multiindex.dtypes
        case pd.Index() as index:
            shape = index.shape
            columns = index.name
            dtypes = index.dtype
        case pd.Series() as series:
            shape = series.shape
            columns = series.name
            dtypes = series.dtype
        case DataFrame() as df:  # type: ignore[misc]
            shape = df.shape  # type: ignore[unreachable]
            columns = df.columns.tolist()
            dtypes = df.dtypes.tolist()
        case pa.Table() as arrow_table:
            shape = arrow_table.shape
            columns = arrow_table.schema.names
            dtypes = arrow_table.schema.types
        case np.ndarray() as arr:
            shape = arr.shape
            columns = None
            dtypes = [arr.dtype]
        case _:
            raise NotImplementedError(
                f"Cannot validate schema for {type(table)} objects!"
            )

    reference_shape, reference_columns, reference_dtypes = reference_schema

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


def validate_object_hash(
    obj: Any,
    /,
    reference: Any,
    *,
    hash_algorithm: Optional[str] = None,
    logger: logging.Logger = __logger__,
    **hash_kwargs: Any,
) -> None:
    """Validate object(s) given reference hash values."""
    # Try to determine the hash algorithm from the array type
    match obj:
        case str() | Path():
            validate_file_hash(
                obj,
                reference,
                hash_algorithm=hash_algorithm,
                logger=logger,
                **hash_kwargs,
            )
        case Table():
            validate_table_hash(
                obj,
                reference,
                hash_algorithm=hash_algorithm,
                logger=logger,
                **hash_kwargs,
            )
        case Mapping():  # apply recursively to all values
            if not isinstance(reference, None | Mapping):
                raise ValueError(
                    "If a mapping of objects is provided, reference_hash must be a mapping as well!"
                )
            for key, value in obj.items():
                validate_object_hash(
                    value,
                    reference.get(key, None),
                    hash_algorithm=hash_algorithm,
                    logger=logger,
                    **hash_kwargs,
                )
        case Iterable():
            match reference:
                case str():
                    raise ValueError("Cannot multiple items with single hash!")
                case None:
                    # apply recursively to all values
                    for value in obj:
                        validate_object_hash(
                            value,
                            reference,
                            hash_algorithm=hash_algorithm,
                            logger=logger,
                            **hash_kwargs,
                        )
                case Iterable():
                    for value, ref in zip(obj, reference, strict=True):
                        validate_object_hash(
                            value,
                            ref,
                            hash_algorithm=hash_algorithm,
                            logger=logger,
                            **hash_kwargs,
                        )
                case _:
                    raise ValueError(f"Invalid reference hash type! {reference=}")
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
                    f"Ignore this warning if the format is parquet.",
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
