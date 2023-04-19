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
]

import hashlib
import logging
import string
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas
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

    row_hash = pandas.util.hash_pandas_object(df, index=index)

    if row_invariant:
        hash_value = hash_set(row_hash, ignore_duplicates=ignore_duplicates)
    else:
        hash_value = hash(tuple(row_hash))

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


def validate_file(
    file: PathLike,
    reference_hash: str | None = None,
    *,
    hash_algorithm: str | None = None,
    hash_table: Mapping[str, tuple[str, str]] | None = None,
    logger: logging.Logger = __logger__,
    **hash_kwargs: Any,
) -> None:
    """Validate the file hash, given hash values from a table."""
    file = Path(file)

    if hash_table is not None and reference_hash is not None:
        raise ValueError("Provide either reference_hash or hash_table, not both.")

    if hash_table is not None:
        hash_algorithm, reference_hash = hash_table.get(
            str(file),
            hash_table.get(file.name, hash_table.get(file.stem, (None, None))),
        )  # try to match by full path, then by name, then by stem

    hash_algorithm = "sha256" if hash_algorithm is None else hash_algorithm
    hash_value = hash_file(file, hash_algorithm=hash_algorithm, **hash_kwargs)

    if reference_hash is None:
        warnings.warn(
            f"No reference hash given for file {file.name!r}."
            f"The {hash_algorithm!r}-hash is {hash_value!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif hash_value != reference_hash:
        warnings.warn(
            f"File {file.name!r} failed to validate!"
            f"File hash {hash_value!r} does not match reference {reference_hash!r}."
            f"Ignore this warning if the format is parquet.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        logger.info(
            "File '%s' validated successfully with '%s'-hash '%s'.",
            file,
            hash_algorithm,
            hash_value,
        )


def validatate_pandas(
    df: DataFrame,
    reference_hash: str | None = None,
    *,
    hash_algorithm: str | None = None,
    hash_table: Mapping[str, tuple[str, str]] | None = None,
    logger: logging.Logger = __logger__,
    **hash_kwargs: Any,
) -> None:
    """Validate the hash of a pandas object, given hash values from a table."""
    if hash_table is not None and reference_hash is not None:
        raise ValueError("Provide either reference_hash or hash_table, not both.")

    if not hasattr(df, "name"):
        raise ValueError("Pandas object must have a name attribute.")

    if hash_table is not None:
        hash_algorithm, reference_hash = hash_table.get(
            df.name,
            hash_table.get(df.name, hash_table.get(df.name, (None, None))),
        )  # try to match by full path, then by name, then by stem

    hash_algorithm = "sha256" if hash_algorithm is None else hash_algorithm
    hash_value = hash_pandas(df, hash_algorithm=hash_algorithm, **hash_kwargs)

    if reference_hash is None:
        warnings.warn(
            f"No reference hash given for pandas object {df.name!r}."
            f"The {hash_algorithm!r}-hash is {hash_value!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif hash_value != reference_hash:
        warnings.warn(
            f"Pandas object {df.name!r} failed to validate!"
            f"Hash {hash_value!r} does not match reference {reference_hash!r}."
            f"Ignore this warning if the format is parquet.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        logger.info(
            "Pandas object '%s' validated successfully with '%s'-hash '%s'.",
            df,
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
