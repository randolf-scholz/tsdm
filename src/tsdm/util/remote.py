r"""Implements a downloader for the TSDM-package."""

__all__ = [
    "hash_file",
    "download",
    "validate_hash",
]

import hashlib
from pathlib import Path
from typing import Any

import requests
from tqdm.autonotebook import tqdm


def hash_file(
    file: str, block_size: int = 65536, algorithm: str = "sha256", **kwargs: Any
) -> str:
    r"""Calculate the SHA256-hash of a file."""
    algorithms = vars(hashlib)
    hash_value = algorithms[algorithm](**kwargs)

    with open(file, "rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(block_size), b""):
            hash_value.update(byte_block)

    return hash_value.hexdigest()


def download(url: str, fname: str, chunk_size: int = 1024) -> None:
    r"""Download a file from a URL."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    path = Path(fname)
    try:
        with open(path, "wb") as file, tqdm(
            desc=path,
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
            f"Error occurred while downloading {fname}, deleting partial download."
        ) from e


def validate_hash(fname: str, hash_value: str, hash_type: str = "sha256") -> bool:
    r"""Validate a file against a hash value."""
    return hash_file(fname, algorithm=hash_type) == hash_value
