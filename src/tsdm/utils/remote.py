r"""Implements a downloader for the TSDM-package."""

__all__ = [
    "download",
    "hash_file",
    "import_from_url",
    "validate_hash",
]

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from tqdm.autonotebook import tqdm

from tsdm.types.aliases import PathLike

__logger__ = logging.getLogger(__name__)


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
    skip_existing: bool = False,
    hash_value: Optional[str] = None,
    hash_algorithm: str = "sha256",
    hash_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    r"""Download a file from a URL."""
    hash_kwargs = {} if hash_kwargs is None else hash_kwargs
    response = requests.get(url, stream=True, timeout=10)
    total = int(response.headers.get("content-length", 0))
    path = Path(fname if fname is not None else url.split("/")[-1])

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and path.exists():
        if hash_value is not None:
            validate_hash(
                path, hash_value, hash_algorithm=hash_algorithm, **hash_kwargs
            )
        return

    try:
        with open(path, "wb") as file, tqdm(
            desc=str(path),
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=chunk_size):
                if data:  # filter out keep-alive new chunks
                    size = file.write(data)
                    progress_bar.update(size)
    except Exception as e:
        path.unlink()
        raise RuntimeError(
            f"Error {e!r} occurred while downloading {fname}, deleting partial files."
        ) from e

    if hash_value is not None:
        validate_hash(path, hash_value, hash_algorithm=hash_algorithm, **hash_kwargs)


def import_from_url(
    url: str, fname: Optional[PathLike] = None, *args: Any, **kwargs: Any
) -> None:
    """Wrap download so that it works with Kaggle and GitHub."""
    parsed_url = urlparse(url)
    path = Path(fname if fname is not None else url.split("/")[-1])
    __logger__.info("Downloading %s to %s", url, path)

    if parsed_url.netloc == "www.kaggle.com":
        kaggle_name = Path(parsed_url.path).name
        subprocess.run(
            f"kaggle competitions download -p {str(path)} -c {kaggle_name}",
            shell=True,
            check=True,
        )
    elif parsed_url.netloc == "github.com":
        subprocess.run(
            f"svn export --force {url.replace('tree/main', 'trunk')} {str(path)}",
            shell=True,
            check=True,
        )
    else:  # default parsing, including for UCI dataset
        download(url, path, *args, **kwargs)


# def shorthash(inputs) -> str:
#     r"""Roughly good for 2¹⁶=65536 items."""
#     encoded = json.dumps(dictionary, sort_keys=True).encode()
#
#     return shake_256(inputs).hexdigest(8)


# def validate(
#     self,
#     filespec: Nested[str | Path],
#     /,
#     *,
#     reference: Optional[str | Mapping[str, str]] = None,
# ) -> None:
#     r"""Validate the file hash."""
#     self.LOGGER.debug("Starting to validate dataset")
#
#     if isinstance(filespec, Mapping):
#         for value in filespec.values():
#             self.validate(value, reference=reference)
#         return
#
#     if isinstance(filespec, Sequence) and not isinstance(filespec, (str, Path)):
#         for value in filespec:
#             self.validate(value, reference=reference)
#         return
#
#     assert isinstance(filespec, (str, Path)), f"{filespec=} wrong type!"
#     file = Path(filespec)
#
#     if not file.exists():
#         raise FileNotFoundError(f"File {file.name!r} does not exist!")
#
#     filehash = sha256(file.read_bytes()).hexdigest()
#
#     if reference is None:
#         warnings.warn(
#             f"File {file.name!r} cannot be validated as no hash is stored in {self.__class__}."
#             f"The filehash is {filehash!r}."
#         )
#
#     elif isinstance(reference, str):
#         if filehash != reference:
#             warnings.warn(
#                 f"File {file.name!r} failed to validate!"
#                 f"File hash {filehash!r} does not match reference {reference!r}."
#                 f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
#             )
#         self.LOGGER.info(
#             f"File {file.name!r} validated successfully {filehash=!r}."
#         )
#
#     elif isinstance(reference, Mapping):
#         if not (file.name in reference) ^ (file.stem in reference):
#             warnings.warn(
#                 f"File {file.name!r} cannot be validated as it is not contained in {reference}."
#                 f"The filehash is {filehash!r}."
#                 f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
#             )
#         elif file.name in reference and filehash != reference[file.name]:
#             warnings.warn(
#                 f"File {file.name!r} failed to validate!"
#                 f"File hash {filehash!r} does not match reference {reference[file.name]!r}."
#                 f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
#             )
#         elif file.stem in reference and filehash != reference[file.stem]:
#             warnings.warn(
#                 f"File {file.name!r} failed to validate!"
#                 f"File hash {filehash!r} does not match reference {reference[file.stem]!r}."
#                 f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
#             )
#         else:
#             self.LOGGER.info(
#                 f"File {file.name!r} validated successfully {filehash=!r}."
#             )
#     else:
#         raise TypeError(f"Unsupported type for {reference=}.")
#
#     self.LOGGER.debug("Finished validating file.")
