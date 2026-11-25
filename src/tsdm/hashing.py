r"""Hash function utils."""

__all__ = [
    # Constants
    "HASH_REGEX",
    # Protocols
    "Hasher",
    # Classes
    "Hash",
    # Functions
    "hash_file",
    "hash_zip_contents",
]

import hashlib
import re
from collections.abc import Buffer
from os import PathLike
from typing import Final, NamedTuple, Protocol, overload
from zipfile import ZipFile

from .types import FilePath, FileStream

HASH_REGEX: Final[re.Pattern] = re.compile(
    r"^(?:(?P<alg>[\w-]+):)?(?P<value>[0-9A-Za-z]+)$"
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


def hash_file(
    file: FilePath | FileStream,
    hasher: str | Hasher = "sha256",
    /,
    *,
    block_size: int = 65536,
) -> Hash:
    r"""Hash a file or seekable binary stream, using SHA-256 by default.

    Read from the beginning in blocks of `block_size` bytes, which must be positive.
    Restore a caller-owned stream's original position even if hashing fails.
    A supplied hasher is updated in place, including any existing state.

    Returns:
        The hexadecimal digest and algorithm name.

    Raises:
        ValueError: If `block_size` is not positive.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    hasher = hashlib.new(hasher) if isinstance(hasher, str) else hasher

    match file:
        case str() | PathLike():
            with open(file, "rb") as stream:
                while block := stream.read(block_size):
                    hasher.update(block)

        case stream:
            position = stream.tell()
            try:
                stream.seek(0)
                while block := stream.read(block_size):
                    hasher.update(block)
            finally:
                stream.seek(position)

    return Hash(hasher.hexdigest(), hasher.name)


def hash_zip_contents(
    file: FilePath | FileStream,
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

    Returns:
        The uppercase aggregate digest and the name of the hash function used.

    Raises:
        ValueError: If `chunk_size` is not positive or member names are duplicated.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    hasher = hashlib.new(hasher) if isinstance(hasher, str) else hasher
    algorithm = hasher.name
    item_hashes: dict[str, bytes] = {}
    with ZipFile(file, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if name in item_hashes:
                raise ValueError(f"Duplicate ZIP member name: {name!r}.")

            member_hasher = hashlib.new(algorithm)
            with zf.open(info, "r") as member:
                while chunk := member.read(chunk_size):
                    member_hasher.update(chunk)

            # Preserve the legacy digest of (name, (hexdigest, algorithm)).
            digest_record = f"'{member_hasher.hexdigest()}''{member_hasher.name}'"
            record_hash = hashlib.new(algorithm, digest_record.encode()).digest()
            item_hashes[name] = hashlib.new(
                algorithm, f"'{name}'".encode() + record_hash
            ).digest()

    archive_hasher = hashlib.new(algorithm)
    for item_hash in sorted(set(item_hashes.values())):
        archive_hasher.update(item_hash)
    return Hash(archive_hasher.hexdigest().upper(), archive_hasher.name)
