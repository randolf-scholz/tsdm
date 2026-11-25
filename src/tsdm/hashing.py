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
    "hash_zip_contents",
    "to_alphanumeric",
    "to_base",
]

import hashlib
import re
import string
from collections.abc import Buffer
from typing import Final, NamedTuple, Protocol, overload
from zipfile import ZipFile

from .types import FilePath, FileStream

HEXDIGITS = "0123456789ABCDEF"
r"""Uppercase hexadecimal digits."""

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
                match = HASH_REGEX.match(s)
                if match is None or match.pos != 0 or match.endpos != len(s):
                    raise ValueError(f"Invalid hash value: {s!r}")
                return cls(match.group("value"), match.group("alg"))
            case _:
                raise TypeError(f"Cannot create Hash from {type(arg)}.")

    def to_int(self, *, base: int | None = None) -> int:
        r"""Convert the stored hash value to an integer.

        If `base` is omitted, try bases 2, 8, 10, 16, and 36 in that order,
        selecting the first whose alphabet contains every character. Base-36
        autodetection accepts lowercase letters only.
        """
        value = self.hash_value
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

    def __str__(self) -> str:
        if self.hash_algorithm is None:
            return self.hash_value
        return f"{self.hash_algorithm}:{self.hash_value}"


def to_alphanumeric(n: int, /, *, chars: str = HEXDIGITS) -> str:
    r"""Encode a signed 64-bit integer using a configurable digit alphabet.

    Add $2⁶³$ to shift the signed range to the unsigned 64-bit range, then encode
    in base `len(chars)`. Left-pad with `chars[0]` to the width required by
    $2⁶⁴ - 1$. The default alphabet produces hexadecimal codes.

    Args:
        n: Integer in the range $[-2⁶³, 2⁶³ - 1]$.
        chars: Alphabet of at least two distinct characters, ordered by digit value.
    """
    # int64  range: [-2⁶³, 2⁶³-1] = [-9,223,372,036,854,775,808, +9,223,372,036,854,775,807]
    # uint64 range: [0,    2⁶⁴-1] = [0, 18446744073709551615]
    digits = to_base(n + 2**63, len(chars))
    max_digits = to_base(2**64 - 1, len(chars))
    # pad with leading zeros
    digits = [0] * (len(max_digits) - len(digits)) + digits
    return "".join(chars[i] for i in digits)


def to_base(n: int, base: int, /) -> list[int]:
    r"""Return the most significant digit first in an integer's base representation.

    The base must be at least 2. Zero is represented by `[0]`. The result
    satisfies `n = sum(d * base**k for k, d in enumerate(reversed(digits)))`.

    Raises:
        ValueError: If `n` is negative.

    References:
        - [Converting an integer to another base](https://stackoverflow.com/a/28666223).
    """
    if n < 0:
        raise ValueError("n must be non-negative!")

    digits = []
    while n:
        n, d = divmod(n, base)
        digits.append(d)
    return digits[::-1] or [0]


def hash_file(
    filepath: FilePath,
    hasher: str | Hasher = "sha256",
    /,
    *,
    block_size: int = 65536,
) -> Hash:
    r"""Hash a file's bytes, using SHA-256 by default.

    Read the file in blocks of `block_size` bytes, which must be positive.
    A supplied hasher is updated in place, including any existing state.

    Returns:
        The hexadecimal digest and algorithm name.

    Raises:
        ValueError: If `block_size` is not positive.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    hasher = hashlib.new(hasher) if isinstance(hasher, str) else hasher

    with open(filepath, "rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(block_size), b""):
            hasher.update(byte_block)

    hash_value = hasher.hexdigest()
    return Hash(hash_value, hasher.name)


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
    return Hash(hash_value=archive_hasher.hexdigest().upper(), hash_algorithm="zip")
