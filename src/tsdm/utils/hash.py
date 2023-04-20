r"""Hash function utils."""

__all__ = [
    # Functions
    "hash_file",
    "hash_iterable",
    "hash_mapping",
    "hash_object",
    "hash_pandas",
    "hash_set",
    "to_alphanumeric",
    "to_base",
    "validate_file",
    "validate_array",
]

import hashlib
import logging
import string
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex, Series

from tsdm.types.aliases import PathLike

__logger__ = logging.getLogger(__name__)


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
    if isinstance(x, DataFrame | Series):
        return hash_pandas(x)
    if isinstance(x, Index | MultiIndex):
        return hash_pandas(x.to_frame())
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


def validate_file(
    file: PathLike | Mapping[str, PathLike],
    reference_hash: None | str | Mapping[str, tuple[str, str]] = None,
    *,
    hash_algorithm: str | None = None,
    logger: logging.Logger = __logger__,
    errors: Literal["warn", "raise", "ignore"] = "warn",
    **hash_kwargs: Any,
) -> None:
    """Validate file(s), given reference hash value(s).

    Arguments:
        file: The file to validate. If a mapping is given, every file in the mapping is validated.
        reference_hash: The reference hash value, or a mapping from file names to
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

    if isinstance(file, Mapping):
        for key, value in file.items():
            if isinstance(reference_hash, Mapping):
                alg, ref = reference_hash.get(key, (None, None))
                validate_file(
                    value,
                    reference_hash=ref,
                    hash_algorithm=alg,
                    logger=logger,
                    errors=errors,
                    **hash_kwargs,
                )
            elif reference_hash is None:
                validate_file(
                    value,
                    reference_hash=reference_hash,
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

    # code for single path
    file = Path(file)

    if file.suffix == ".parquet":
        warnings.warn(
            f"Parquet file {file.name!r} not validated since the format is not binary stable!",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(reference_hash, Mapping):
        hash_table = reference_hash
        hash_algorithm, reference_hash = hash_table.get(
            str(file),
            hash_table.get(file.name, hash_table.get(file.stem, (None, None))),
        )  # try to match by full path, then by name, then by stem

    hash_algorithm = "sha256" if hash_algorithm is None else hash_algorithm
    hash_value = hash_file(file, hash_algorithm=hash_algorithm, **hash_kwargs)

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
            f"File {file.name!r} failed to validate!"
            f"File hash {hash_value!r} does not match reference {reference_hash!r}."
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


def validate_array(
    array: Any,
    reference_hash: str | None,
    *,
    hash_algorithm: Optional[str] = None,
    logger: logging.Logger = __logger__,
    **hash_kwargs: Any,
) -> None:
    """Validate the hash of a pandas object, given hash values from a table."""
    # Try to determine the hash algorithm from the array type
    if hash_algorithm is None:
        match array:
            case pd.DataFrame() | pd.Series() | pd.Index():
                hash_algorithm = "pandas"
            case np.ndarray():
                hash_algorithm = "numpy"
            case _:
                raise ValueError(
                    f"Cannot determine hash algorithm for array-like object {array!r}."
                    "Please specify the hash algorithm manually."
                )

    hash_value = hash_array(array, hash_algorithm=hash_algorithm, **hash_kwargs)
    name = f"{type(array)}<{array.shape}> {id(array)}"
    if reference_hash is None:
        warnings.warn(
            f"No reference hash given for array-like object {name!r}."
            f"The {hash_algorithm!r}-hash is {hash_value!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif hash_value != reference_hash:
        warnings.warn(
            f"Array-Like  object {name!r} failed to validate!"
            f"Hash {hash_value!r} does not match reference {reference_hash!r}."
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


# def validate_file(
#     fname: PathLike, hash_value: str, *, hash_algorithm: str = "sha256", **kwargs: Any
# ) -> None:
#     r"""Validate a file against a hash value."""
#     hashed = hash_file(fname, hash_algorithm=hash_algorithm, **kwargs)
#     if hashed != hash_value:
#         raise ValueError(
#             f"Calculated hash value {hashed!r} and given hash value {hash_value!r} "
#             f"do not match for given file {fname!r} (using {hash_algorithm})."
#         )
