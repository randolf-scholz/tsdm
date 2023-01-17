r"""Implements a downloader for the TSDM-package."""

__all__ = [
    "hash_file",
    "download",
    "validate_hash",
]

import hashlib
import string
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm.autonotebook import tqdm

from tsdm.types.aliases import PathLike


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


def validate_hash(
    fname: PathLike, hash_value: str, *, hash_algorithm: str = "sha256", **kwargs: Any
) -> None:
    r"""Validate a file against a hash value."""
    hashed = hash_file(fname, hash_algorithm=hash_algorithm, **kwargs)
    if hashed != hash_value:
        raise ValueError(
            f"Calculated hash value {hashed!r} and given hash value {hash_value!r} "
            f"do not match for given file {fname!r} (using {hash_algorithm})."
        )


def download(
    url: str,
    fname: Optional[PathLike] = None,
    *,
    chunk_size: int = 1024,
    skip_existing: bool = True,
    hash_value: Optional[str] = None,
    hash_algorithm: str = "sha256",
    hash_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    r"""Download a file from a URL."""
    hash_kwargs = {} if hash_kwargs is None else hash_kwargs
    response = requests.get(url, stream=True, timeout=10)
    total = int(response.headers.get("content-length", 0))
    path = Path(fname if fname is not None else url.split("/")[-1])

    if skip_existing and path.exists():
        if hash_value is not None:
            validate_hash(
                path, hash_value, hash_algorithm=hash_algorithm, **hash_kwargs
            )
        return

    try:
        with path.open("wb") as file, tqdm(
            desc=str(path),
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                progress_bar.update(size)
    except Exception as e:
        path.unlink()
        raise RuntimeError(
            f"Error {e!r} occurred while downloading {fname}, deleting partial files."
        ) from e

    if hash_value is not None:
        validate_hash(path, hash_value, hash_algorithm=hash_algorithm, **hash_kwargs)


def to_base(n: int, b: int) -> list[int]:
    r"""Convert non-negative integer to any basis.

    The result satisfies: ``n = sum(d*b**k for k, d in enumerate(reversed(digits)))``

    References
    ----------
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


# def shorthash(inputs) -> str:
#     r"""Roughly good for 2¹⁶=65536 items."""
#     encoded = json.dumps(dictionary, sort_keys=True).encode()
#
#     return shake_256(inputs).hexdigest(8)
