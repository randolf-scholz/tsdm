r"""Experimental hashing for Python objects, arrays, and tables."""

__all__ = [
    "hash_array",
    "tokenize_array",
    "tokenize_bool",
    "tokenize_byte_stream",
    "tokenize_bytes",
    "tokenize_collection",
    "tokenize_complex",
    "tokenize_dataclass",
    "tokenize_datetime",
    "tokenize_ellipsis",
    "tokenize_float",
    "tokenize_int",
    "tokenize_mapping",
    "tokenize_multiset",
    "tokenize_none",
    "tokenize_notimplemented",
    "tokenize_numpy",
    "tokenize_object",
    "tokenize_pandas",
    "tokenize_polars",
    "tokenize_pyarrow",
    "tokenize_set",
    "tokenize_str",
    "tokenize_timedelta",
    "tokenize_type",
]

import dataclasses
import datetime as dt
import hashlib
import math
import re
import struct
from collections import Counter
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    Set as AbstractSet,
)
from types import EllipsisType, NoneType, NotImplementedType
from typing import Any, Final, NamedTuple, Protocol, overload

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from numpy.typing import NDArray

from tsdm.types import DataclassType, SupportsArray

HASH_REGEX: Final = re.compile(r"^(?:(?P<alg>[\w-]+):)?(?P<value>[0-9A-Za-z]+)$")
r"""Regular expression to match hash values."""


class Hasher(Protocol):
    r"""Protocol for hash functions."""

    @property
    def name(self) -> str: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...


class Hash(NamedTuple):
    r"""Hash value with optional algorithm."""

    hash_value: str
    hash_algorithm: str | None

    @overload
    @classmethod
    def from_value(cls, arg: None, /) -> None: ...
    @overload
    @classmethod
    def from_value(cls, arg: str | bytes | Hash, /) -> Hash: ...
    @classmethod
    def from_value(cls, arg: str | bytes | Hash | None, /) -> Hash | None:
        r"""Parse a hash string or encode digest bytes as hexadecimal.

        Strings have the form `value` or `algorithm:value`, with an alphanumeric
        value. Existing `Hash` objects and `None` are returned unchanged.

        Raises:
            ValueError: If a string does not match the hash syntax.
            TypeError: If the argument has an unsupported type.
        """
        match arg:
            case None:
                return None
            case Hash():
                return arg
            case bytes():
                return cls(arg.hex(), None)
            case str(s):
                match = HASH_REGEX.fullmatch(s)
                if match is None:
                    raise ValueError(f"Invalid hash value: {s!r}")
                return cls(match.group("value"), match.group("alg"))
            case _:
                raise TypeError(f"Cannot create Hash from {type(arg)}.")

    def __str__(self) -> str:
        if self.hash_algorithm is None:
            return self.hash_value
        return f"{self.hash_algorithm}:{self.hash_value}"


def tokenize_type(arg: type, /) -> bytes:
    return f"{arg.__module__}.{arg.__qualname__}".encode()


def tokenize_none(_: None, /) -> bytes:
    return b"None"


def tokenize_ellipsis(_: EllipsisType, /) -> bytes:
    return b"..."


def tokenize_notimplemented(_: NotImplementedType, /) -> bytes:
    return b"NotImplemented"


def tokenize_bool(arg: bool, /) -> bytes:  # ruff: ignore[FBT001]
    return b"\x01" if arg else b"\x00"


def tokenize_int(arg: int, /) -> bytes:
    num_bytes = arg.bit_length() // 8 + 1
    return arg.to_bytes(num_bytes, "little", signed=True)


def tokenize_float(arg: float, /) -> bytes:
    # Canonicalize edge cases
    if math.isnan(arg):  # fixed NaN payload
        return b"\x7f\xf8\x00\x00\x00\x00\x00\x00"  # IEEE-754 binary64 NaN
    if arg == 0.0:  # normalize -0.0 to +0.0
        return struct.pack(">d", 0.0)
    return struct.pack(">d", arg)  # IEEE-754 binary64, big-endian


def tokenize_complex(arg: complex, /) -> bytes:
    return tokenize_float(arg.real) + b"+" + tokenize_float(arg.imag) + b"j"


def tokenize_str(arg: str, /) -> bytes:
    return b"'" + arg.encode("utf-8") + b"'"


def tokenize_bytes(arg: bytes, /) -> bytes:
    return arg


def tokenize_datetime(arg: dt.datetime | dt.date | dt.time, /) -> bytes:
    return arg.isoformat().encode("utf-8")


def tokenize_timedelta(arg: dt.timedelta, /) -> bytes:
    return (
        tokenize_int(arg.days)
        + b"d"
        + tokenize_int(arg.seconds)
        + b"s"
        + tokenize_int(arg.microseconds)
        + b"us"
    )


_BASIC_TOKENIZERS: dict[type, Callable[[Any], bytes]] = {
    NoneType           : tokenize_none,
    EllipsisType       : tokenize_ellipsis,
    NotImplementedType : tokenize_notimplemented,
    bool               : tokenize_bool,
    int                : tokenize_int,
    float              : tokenize_float,
    complex            : tokenize_complex,
    str                : tokenize_str,
    bytes              : tokenize_bytes,
    dt.datetime        : tokenize_datetime,
    dt.date            : tokenize_datetime,
    dt.time            : tokenize_datetime,
    dt.timedelta       : tokenize_timedelta,
    type               : tokenize_type,
}  # fmt: skip


def tokenize_dataclass(arg: DataclassType, hasher: str | Hasher, /) -> bytes:
    return tokenize_mapping(dataclasses.asdict(arg), hasher)


def tokenize_mapping(arg: Mapping[Any, object], hasher: str | Hasher, /) -> bytes:
    r"""Return digest bytes for a mapping, independent of item order.

    Keys and values must be supported by `tokenize_object`; they need not be
    hashable Python objects.
    """
    return tokenize_set(arg.items(), hasher)


def tokenize_set(arg: Iterable[object], hasher: str | Hasher, /) -> bytes:
    r"""Return digest bytes for an iterable, independent of element order.

    Deduplicate elements by their tokens, discarding multiplicity. Elements must
    be supported by `tokenize_object`.
    """
    set_of_hashes = {tokenize_object(y, hasher) for y in arg}
    return tokenize_collection(sorted(set_of_hashes), hasher)


def tokenize_multiset(arg: Iterable[object], hasher: str | Hasher, /) -> bytes:
    r"""Return digest bytes for an iterable, independent of element order.

    Preserve the multiplicity of element tokens. Elements must be supported by
    `tokenize_object`.
    """
    # We use counter as a proxy for multisets, do deal with duplicates in x.
    mdict: Counter[bytes] = Counter(tokenize_object(y, hasher) for y in arg)
    return tokenize_collection(sorted(mdict.elements()), hasher)


def tokenize_collection(arg: Iterable[object], hasher: str | Hasher, /) -> bytes:
    r"""Return digest bytes for supported iterable elements in their supplied order."""
    hash_algorithm = hasher if isinstance(hasher, str) else hasher.name
    byte_stream = (tokenize_object(item, hashlib.new(hash_algorithm)) for item in arg)
    return tokenize_byte_stream(byte_stream, hasher)


def tokenize_byte_stream(arg: Iterable[bytes], hasher: str | Hasher, /) -> bytes:
    r"""Hash concatenated byte chunks in order using a fresh hasher.

    Chunk boundaries do not affect the digest. A supplied hasher contributes its
    algorithm name only; its state is neither used nor modified.
    """
    # NOTE: This is the only place the hasher actually gets used.
    hash_algorithm = hasher if isinstance(hasher, str) else hasher.name
    collection_hasher = hashlib.new(hash_algorithm)
    for item in arg:
        collection_hasher.update(item)
    return collection_hasher.digest()


def tokenize_numpy(arg: NDArray, hasher: str | Hasher, /) -> bytes:
    r"""Hash a numpy array."""
    shape = list(arg.shape)
    items = np.asarray(arg).flatten().tolist()
    return tokenize_collection(shape + items, hasher)


def tokenize_pandas(
    arg: pd.Index | pd.Series | pd.DataFrame,
    hasher: str | Hasher,
    /,
    *,
    index: bool = True,
) -> bytes:
    r"""Return digest bytes for a pandas Index, Series, or DataFrame.

    Index and Series values are tokenized in order. For a Series, `index=True`
    includes the index labels by tokenizing label-value pairs.

    A DataFrame is converted to a nested dictionary and tokenized as a mapping,
    so mapping item order does not affect the digest. The `index` option is
    forwarded to `DataFrame.to_dict` and is subject to that method's restrictions.

    Raises:
        TypeError: If the object is not a supported pandas type.
    """
    match arg:
        case pd.Series():
            return tokenize_collection(arg.items() if index else arg, hasher)
        case pd.Index():
            return tokenize_collection(arg, hasher)
        case pd.DataFrame():
            return tokenize_mapping(arg.to_dict(index=index), hasher)
        case _:
            raise TypeError(f"Cannot hash pandas object of type {type(arg)}.")


def tokenize_polars(arg: pl.DataFrame | pl.Series, hasher: str | Hasher, /) -> bytes:
    r"""Hash a polars DataFrame or Series."""
    match arg:
        case pl.Series():
            return tokenize_collection(arg, hasher)
        case pl.DataFrame():
            return tokenize_mapping(arg.to_dict(as_series=True), hasher)
        case _:
            raise TypeError(f"Cannot hash polars object of type {type(arg)}.")


def tokenize_pyarrow(
    arg: pa.Scalar | pa.Array | pa.Table, hasher: str | Hasher, /
) -> bytes:
    r"""Return digest bytes for a PyArrow scalar, array, chunked array, or table."""
    match arg:
        case pa.Scalar():
            return tokenize_object(arg.as_py(), hasher)
        case pa.Array() | pa.ChunkedArray():
            return tokenize_collection(arg.to_pylist(), hasher)
        case pa.Table():
            return tokenize_mapping(arg.to_pydict(), hasher)
        case _:
            raise TypeError(f"Cannot hash pyarrow object of type {type(arg)}.")


def tokenize_array(array: SupportsArray, hasher: str | Hasher, /) -> bytes:
    r"""Return digest bytes for a supported array or table.

    Dispatch to the pandas, PyArrow, or Polars tokenizer when applicable.
    Otherwise, convert objects implementing `__array__` to NumPy arrays.
    """
    hasher = hashlib.new(hasher) if isinstance(hasher, str) else hasher

    match array:
        case pd.DataFrame() | pd.Series() | pd.Index():
            return tokenize_pandas(array, hasher)
        case pa.Table() | pa.Array() | pa.ChunkedArray():
            return tokenize_pyarrow(array, hasher)
        case pl.DataFrame() | pl.Series():
            return tokenize_polars(array, hasher)
        case SupportsArray():
            return tokenize_numpy(array.__array__(), hasher)
        case _:
            raise TypeError(f"Cannot hash array of type {type(array)}.")


def tokenize_object(x: object, hasher: str | Hasher, /) -> bytes:
    r"""Hash an object."""
    kind = type(x)
    hash_func = _BASIC_TOKENIZERS.get(kind)

    match x:
        case basic_type if hash_func is not None:
            return hash_func(basic_type)
        case SupportsArray():
            return tokenize_array(x, hasher)
        case DataclassType():
            return tokenize_dataclass(x, hasher)
        case Mapping():
            return tokenize_mapping(x, hasher)
        case AbstractSet():  # set / frozenset
            return tokenize_set(x, hasher)
        case Collection():  # list / tuple
            return tokenize_collection(x, hasher)
        case _:
            raise TypeError(f"Cannot hash object of type {type(x)}.")


def hash_array(array: SupportsArray, hasher: str | Hasher, /) -> Hash:
    r"""Return a hexadecimal hash and algorithm name for a supported array or table."""
    hasher = hashlib.new(hasher if isinstance(hasher, str) else hasher.name)
    hash_value = tokenize_array(array, hasher).hex()
    return Hash(hash_value, hasher.name)
