r"""Hash function utils."""

__all__ = [
    # Constants
    "HASH_REGEX",
    # Classes
    "Hash",
    # Functions
    "hash_file",
    "hash_zip_contents",
]

import hashlib
import re
from os import PathLike
from typing import Final, NamedTuple, overload
from zipfile import ZipFile

from .types import FilePath, FileStream

HASH_REGEX: Final = re.compile(r"^(?:(?P<alg>[\w-]+):)?(?P<value>[0-9A-Za-z]+)$")
r"""Regular expression to match hash values."""


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


def hash_file(file: FilePath | FileStream, /, algorithm: str = "sha256") -> Hash:
    r"""Hash a file or seekable binary stream, using SHA-256 by default.

    Hash from the beginning with `hashlib.file_digest`. Restore a caller-owned
    stream's original position even if hashing fails.

    Args:
        file: A path or a seekable binary stream supported by `hashlib.file_digest`.
        algorithm: Name of a fixed-output hash function supported by `hashlib`.

    Returns:
        The lowercase hexadecimal digest and algorithm name.
    """
    match file:
        case str() | PathLike():
            with open(file, "rb") as stream:
                hasher = hashlib.file_digest(stream, algorithm)

        case stream:
            position = stream.tell()
            try:
                stream.seek(0)
                hasher = hashlib.file_digest(stream, algorithm)  # type: ignore  # lacks getbuffer()
            finally:
                stream.seek(position)

    return Hash(hasher.hexdigest(), hasher.name)


def hash_zip_contents(
    file: FilePath | FileStream, /, algorithm: str = "sha256"
) -> Hash:
    r"""Hash the aggregate contents of a ZIP archive.

    For each member in filename order, feed two raw digests into the archive
    hash: the digest of its UTF-8 name and then the digest of its uncompressed
    contents. All steps use the same fixed-output hash function. The checksum
    is independent of ZIP metadata such as timestamps, compression, and member
    order.

    Args:
        file: A ZIP archive path or a seekable binary stream.
        algorithm: Name of a fixed-output hash function supported by `hashlib`.

    Returns:
        The lowercase hexadecimal aggregate digest and algorithm name.

    Raises:
        ValueError: If member names are duplicated.
    """
    archive_hasher = hashlib.new(algorithm)

    with ZipFile(file, "r") as zf:
        members = sorted(zf.infolist(), key=lambda i: i.filename)

        if len({i.filename for i in members}) != len(members):
            raise ValueError(f"Duplicate ZIP member names in {members}")

        for info in members:
            with zf.open(info, "r") as stream:
                content_hash = hashlib.file_digest(stream, algorithm)  # type: ignore  # lacks getbuffer()

            name_tokens = info.filename.encode("utf-8")
            name_hash = hashlib.new(algorithm, name_tokens)
            archive_hasher.update(name_hash.digest())
            archive_hasher.update(content_hash.digest())

    return Hash(archive_hasher.hexdigest(), archive_hasher.name)
