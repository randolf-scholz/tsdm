r"""Hash function utils."""

__all__ = [
    # Constants
    "HASH_REGEX",
    "HEXDIGITS",
    # Protocols
    "Hasher",
    # Classes
    "Hash",
    # Functions
    "hash_file",
    "hash_array",
    "hash_zip_contents",
    "to_alphanumeric",
    "to_base",
    "to_int",
    "tokenize_array",
    "tokenize_bool",
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
import string
import struct
from collections import Counter
from collections.abc import (
    Buffer,
    Callable,
    Collection,
    Hashable,
    Iterable,
    Mapping,
    Set as AbstractSet,
)
from types import EllipsisType, NoneType, NotImplementedType
from typing import (
    IO,
    Any,
    Final,
    NamedTuple,
    Optional,
    Protocol,
    overload,
)
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series

from tsdm.types.aliases import FilePath
from tsdm.types.mixins import SupportsArray
from tsdm.types.protocols import Dataclass

HEXDIGITS = "0123456789ABCDEF"
r"""Upppercase hexadecimal digits."""

HASH_REGEX: Final[re.Pattern] = re.compile(
    r"^(?:(?P<alg>\w+):)?(?P<value>[0-9A-Za-z]+)$"
)
r"""Regular expression to match hash values."""


class Hasher(Protocol):
    r"""Protocol for hash functions."""

    @property
    def name(self) -> str: ...
    def update(self, data: Buffer, /) -> None: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...


class Hash(NamedTuple):
    r"""Hash value with optional algorithm."""

    hash_value: str
    hash_algorithm: str | None

    def __hash__(self) -> int:
        # forward compatibility in case fields are added later.
        return hash((self.hash_algorithm, self.hash_value))

    @overload
    @classmethod
    def from_value(cls, arg: None, /) -> None: ...
    @overload
    @classmethod
    def from_value(cls, arg: str | bytes | Hash, /) -> Hash: ...
    @classmethod
    def from_value(cls, arg: str | bytes | Hash | None, /) -> Hash | None:
        r"""Create a `Hash` from a string or integer value."""
        match arg:
            case None:
                return None
            case Hash():
                return arg
            case bytes():
                return cls(arg.hex(), None)
            case str(s):
                match = HASH_REGEX.match(s)
                if match is None or match.pos != 0 or match.endpos != len(s):
                    raise ValueError(f"Invalid hash value: {s!r}")
                return cls(match.group("value"), match.group("alg"))
            case _:
                raise TypeError(f"Cannot create Hash from {type(arg)}.")

    def to_int(self, *, base: Optional[int] = None) -> int:
        return to_int(self.hash_value, base=base)

    def __str__(self) -> str:
        if self.hash_algorithm is None:
            return self.hash_value
        return f"{self.hash_algorithm}:{self.hash_value}"


def to_int(value: str, /, *, base: Optional[int] = None) -> int:
    r"""Convert a string to an integer, autodetecting the base if necessary."""
    # if only decimals:
    if base is None:
        if all(c in "01" for c in value):  # binary
            base = 2
        elif all(c in string.octdigits for c in value):  # octal
            base = 8
        elif all(c in string.digits for c in value):  # decimal
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


def to_alphanumeric(n: int, /, *, chars: str = HEXDIGITS) -> str:
    r"""Convert integer to alphanumeric code.

    Note:
        We assume n is an int64. We first convert it to uint64, then to base 36.
        Doubling the alphabet size generally reduces the length of the code by
        a factor of $1 + 1/log₂B$, where $B$ is the alphabet size.
    """
    # int64  range: [-2⁶³, 2⁶³-1] = [-9,223,372,036,854,775,808, +9,223,372,036,854,775,807]
    # uint64 range: [0,    2⁶⁴-1] = [0, 18446744073709551615]
    digits = to_base(n + 2**63, len(chars))
    max_digits = to_base(2**64 - 1, len(chars))
    # pad with leading zeros
    digits = [0] * (len(max_digits) - len(digits)) + digits
    return "".join(chars[i] for i in digits)


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


def tokenize_type(arg: type, /) -> bytes:
    return f"{arg.__module__}.{arg.__qualname__}".encode()


def tokenize_none(x: None, /) -> bytes:
    return tokenize_type(type(x))


def tokenize_ellipsis(x: EllipsisType, /) -> bytes:
    return tokenize_type(type(x))


def tokenize_notimplemented(x: NotImplementedType, /) -> bytes:
    return tokenize_type(type(x))


def tokenize_bool(x: bool, /) -> bytes:  # noqa: FBT001
    return b"\x01" if x else b"\x00"


def tokenize_int(x: int, /) -> bytes:
    bits = x.bit_length() // 8 + 1
    return x.to_bytes(bits, "little", signed=True)


def tokenize_float(x: float, /) -> bytes:
    # Canonicalize edge cases
    if math.isnan(x):  # fixed NaN payload
        return b"\x7f\xf8\x00\x00\x00\x00\x00\x00"  # IEEE-754 binary64 NaN
    if x == 0.0:  # normalize -0.0 to +0.0
        return struct.pack(">d", 0.0)
    return struct.pack(">d", x)  # IEEE-754 binary64, big-endian


def tokenize_complex(x: complex, /) -> bytes:
    return tokenize_float(x.real) + tokenize_float(x.imag)


def tokenize_str(arg: str, /) -> bytes:
    return arg.encode("utf-8")


def tokenize_bytes(arg: bytes, /) -> bytes:
    return arg


def tokenize_datetime(arg: dt.datetime | dt.date | dt.time, /) -> bytes:
    return arg.isoformat().encode("utf-8")


def tokenize_timedelta(arg: dt.timedelta, /) -> bytes:
    return tokenize_float(arg.total_seconds())


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


def tokenize_dataclass(arg: Dataclass, hasher: str | Hasher, /) -> bytes:
    return tokenize_mapping(dataclasses.asdict(arg), hasher)


def tokenize_mapping(x: Mapping[Any, Hashable], hasher: str | Hasher, /) -> bytes:
    r"""Hash a `Mapping` of hashable objects in a permutation invariant manner."""
    return tokenize_set(x.items(), hasher)


def tokenize_set(x: Iterable[Hashable], hasher: str | Hasher, /) -> bytes:
    r"""Hash an `Iterable` of hashable objects in a permutation invariant manner.

    Discards multiplicity of elements.
    """
    set_of_hashes = {tokenize_object(y, hasher) for y in x}
    return tokenize_collection(sorted(set_of_hashes), hasher)


def tokenize_multiset(x: Iterable[Hashable], hasher: str | Hasher, /) -> bytes:
    r"""Hash an `Iterable` of hashable objects in a permutation invariant manner.

    Takes multiplicity of elements into account.
    """
    # We use counter as a proxy for multisets, do deal with duplicates in x.
    mdict: Counter[bytes] = Counter(tokenize_object(y, hasher) for y in x)
    return tokenize_collection(sorted(mdict.elements()), hasher)


def tokenize_collection(x: Collection[Hashable], hasher: str | Hasher, /) -> bytes:
    r"""Hash a `Collection` of hashable objects in an order-dependent manner."""
    # NOTE: This is the only place the hasher actually gets used.
    hash_algorithm = hasher if isinstance(hasher, str) else hasher.name
    collection_hasher = hashlib.new(hash_algorithm)
    for item in x:
        # hash each item individually
        item_hasher = hashlib.new(hash_algorithm)
        object_hash = tokenize_object(item, item_hasher)
        collection_hasher.update(object_hash)

    return collection_hasher.digest()


def tokenize_pandas(
    df: Index | Series | DataFrame,
    hasher: str | Hasher,
    /,
    *,
    index: bool = True,
    ignore_duplicates: bool = False,
    row_invariant: bool = False,
    col_invariant: bool = False,
) -> bytes:
    r"""Hash pandas object to a single number.

    If row_invariant is True, then the hash is invariant to the order of the rows.
    If col_invariant is True, then the hash is invariant to the order of the columns.

    .. math:: hash(PX) = hash(X) and hash(XQ) = hash(X)

    for permutation matrices $P$ and $Q$ respectively.
    """
    # TODO: problem: duplicate rows ⇝ include standard index for rows/cols!

    if col_invariant:
        df = df.stack()

    row_hash: Series = pd.util.hash_pandas_object(df, index=index)

    hash_value = (
        tokenize_set(row_hash, hasher)
        if row_invariant and ignore_duplicates
        else tokenize_multiset(row_hash, hasher)
        if row_invariant
        else tokenize_collection(row_hash.tolist(), hasher)
    )

    return hash_value


def tokenize_numpy(array: NDArray, hasher: str | Hasher, /) -> bytes:
    r"""Hash a numpy array."""
    shape = list(array.shape)
    items = np.asarray(array).flatten().tolist()
    return tokenize_collection(shape + items, hasher)


def tokenize_pyarrow(array: pa.Array, hasher: str | Hasher, /) -> bytes:
    r"""Hash a pyarrow array."""
    raise NotImplementedError(f"Can't hash {type(array)} yet.")


def tokenize_array(array: SupportsArray, hasher: str | Hasher, /) -> bytes:
    r"""Hash an array like object (pandas/numpy/pyarrow/etc.)."""
    hasher = hashlib.new(hasher) if isinstance(hasher, str) else hasher

    match array:
        case DataFrame() | Series() | Index():
            return tokenize_pandas(array, hasher)
        case pa.Table() | pa.Array() | pa.ChunkedArray():
            return tokenize_pyarrow(array, hasher)
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
        case Dataclass():
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
    r"""Hash an array like object (pandas/numpy/pyarrow/etc.)."""
    hasher = hashlib.new(hasher if isinstance(hasher, str) else hasher.name)
    hash_value = tokenize_array(array, hasher).hex()
    return Hash(hash_value, hasher.name)


def hash_file(
    filepath: FilePath,
    hasher: str | Hasher = "sha256",
    /,
    *,
    block_size: int = 65536,
) -> Hash:
    r"""Calculate the SHA256-hash of a file."""
    hasher = hashlib.new(hasher) if isinstance(hasher, str) else hasher

    with open(filepath, "rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(block_size), b""):
            hasher.update(byte_block)

    hash_value = hasher.hexdigest()
    return Hash(hash_value, hasher.name)


def hash_zip_contents(
    file: FilePath | IO[bytes],
    hasher: str | Hasher = "sha256",
    /,
    *,
    chunk_size: int = 1024 * 1024,
) -> Hash:
    r"""Hash the *contents* of a zip archive (stable w.r.t. zip metadata).

    The resulting digest depends only on:
    - the member names
    - the uncompressed bytes of each member

    It is *independent* of zip container metadata such as timestamps, compression method,
    extra fields, file order in the central directory, etc.
    """
    hasher = hashlib.new(hasher) if isinstance(hasher, str) else hasher
    item_hashes: dict[str, Hash] = {}
    with ZipFile(file, "r") as zf:
        for name in zf.namelist():
            # make a new hasher for each item
            hasher = hashlib.new(hasher.name)
            with zf.open(name, "r") as member:
                while chunk := member.read(chunk_size):
                    hasher.update(chunk)
            item_hashes[name] = Hash(
                hash_value=hasher.hexdigest(),
                hash_algorithm=hasher.name,
            )
    hash_value = tokenize_set(item_hashes.items(), hasher)
    return Hash(hash_value=hash_value.hex().upper(), hash_algorithm="zip")
