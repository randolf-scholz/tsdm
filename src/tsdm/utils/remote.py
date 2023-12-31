r"""Implements a downloader for the TSDM-package."""

__all__ = [
    "download",
    "download_directory_to_zip",
    "download_io",
    "import_from_url",
    "iter_content",
    "stream_download",
]

import logging
import os
import subprocess
from collections.abc import Iterator, Mapping
from html.parser import HTMLParser
from http import HTTPStatus
from io import IOBase
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import requests
from requests import Session
from tqdm.autonotebook import tqdm
from typing_extensions import IO, Any, Optional

from tsdm.constants import EMPTY_MAP
from tsdm.testing.hash import validate_file_hash
from tsdm.types.aliases import PathLike


class LinkParser(HTMLParser):
    """Parse links from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    self.links.append(value)


def download_io(
    url: str,
    file: IO | IOBase,
    *,
    session: Optional[Session] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Mapping[str, str] = EMPTY_MAP,
    request_options: Mapping[str, Any] = EMPTY_MAP,
    chunk_size: int = 1024,
) -> None:
    """Download a file from a URL to an IO stream."""
    if session is None:
        # construct the request
        request_options = {
            "headers": headers,
            "auth": None if username is None else (username, password),
            "stream": True,
            "timeout": 10,
        } | dict(request_options)
        response = requests.get(url, **request_options)
    else:
        response = session.get(url)

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Failed to download {url} with status code {response.status_code}."
        )

    with tqdm(
        desc=f"Downloading {url}",
        total=int(response.headers.get("content-length", 0)),
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=chunk_size):
            if data:  # filter out keep-alive new chunks
                progress_bar.update(chunk_size)
                file.write(data)


def stream_download(
    url: str,
    *,
    session: Optional[Session] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Mapping[str, str] = EMPTY_MAP,
    request_options: Mapping[str, Any] = EMPTY_MAP,
    chunk_size: int = 1024,
) -> Iterator[bytes]:
    """Download a file as a bytes-stream."""
    if session is None:
        # construct the request
        request_options = {
            "headers": headers,
            "auth": None if username is None else (username, password),
            "stream": True,
            "timeout": 10,
        } | dict(request_options)
        response = requests.get(url, **request_options)
    else:
        response = session.get(url)

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Failed to download {url} with status code {response.status_code}."
        )
    with tqdm(
        desc=f"Downloading {url}",
        total=int(response.headers.get("content-length", 0)),
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=chunk_size):
            if data:  # filter out keep-alive new chunks
                progress_bar.update(chunk_size)
                yield data


def iter_content(url: str, /, *, session: Session) -> Iterator[str]:
    """Iterate over the contents of a directory."""
    response = session.get(url)
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Failed to download {url} with status code {response.status_code}."
        )

    parser = LinkParser()
    parser.feed(response.text)

    for link in parser.links:
        if link == "../":
            continue

        if link.endswith("/"):  # Recursion
            yield from iter_content(url + link, session=session)
        else:
            yield url + link


# NOTE: Session options as of requests 2.26.0
# __attrs__ = [
#     "headers",
#     "cookies",
#     "auth",
#     "proxies",
#     "hooks",
#     "params",
#     "verify",
#     "cert",
#     "adapters",
#     "stream",
#     "trust_env",
#     "max_redirects",
# ]
def download_directory_to_zip(
    url: str,
    zip_filename: PathLike,
    *,
    # session options
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Mapping[str, str] = EMPTY_MAP,
    stream: bool = True,
    add_toplevel_dir: bool = True,
    zip_options: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    """Download a directory from a URL to a zip file."""
    path = Path(zip_filename)
    assert path.suffix == ".zip", f"{path=} must have .zip suffix!"
    stem = f"{path.stem}/" if add_toplevel_dir else ""
    zip_options = {"mode": "w"} | dict(zip_options)

    # Create a session
    with Session() as session:
        session.auth = (
            (username, password)
            if username is not None and password is not None
            else None
        )
        session.headers.update(headers)
        session.stream = stream

        response = session.get(url)
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"Failed to create session for {url} with status code"
                f" {response.status_code}."
            )

        # Get the contents of the directory
        content: list[str] = sorted(iter_content(url, session=session))

        # Download the directory
        with ZipFile(zip_filename, **zip_options) as archive:
            for href in (pbar := tqdm(content)):
                # get relative path w.r.t. the base url
                file_name = os.path.relpath(href, url)
                pbar.set_description(f"Downloading {file_name}")
                # file_size = int(session.head(href).headers.get("content-length", 0))
                with archive.open(stem + file_name, "w", force_zip64=True) as file:
                    download_io(href, file, session=session)


def download(
    url: str,
    fname: Optional[PathLike | IOBase] = None,
    *,
    # request options
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Mapping[str, str] = EMPTY_MAP,
    request_options: Mapping[str, Any] = EMPTY_MAP,
    chunk_size: int = 1024,
    # file options
    skip_existing: bool = False,
    hash_value: Optional[str] = None,
    hash_algorithm: Optional[str] = None,
    hash_kwargs: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    r"""Download a file from a URL.

    This is essentially a wrapper around `requests.get` with a progress bar.
    """
    if isinstance(fname, IOBase):
        download_io(
            url,
            fname,
            username=username,
            password=password,
            headers=headers,
            request_options=request_options,
            chunk_size=chunk_size,
        )
        return

    # construct the path
    path = Path(url.split("/")[-1] if fname is None else fname)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    # check if the file already exists
    if skip_existing and path.exists():
        if hash_value is not None:
            validate_file_hash(
                path, hash_value, hash_algorithm=hash_algorithm, hash_kwargs=hash_kwargs
            )
        return

    # attempt to download the file
    try:
        with open(path, "wb") as file:
            download_io(
                url,
                file,
                username=username,
                password=password,
                headers=headers,
                request_options=request_options,
                chunk_size=chunk_size,
            )
    except Exception as exc:
        path.unlink()
        raise RuntimeError(
            f"Error {exc!r} occurred while downloading {fname}, deleting partial files."
        ) from exc

    # validate the file hash
    if hash_value is not None:
        validate_file_hash(
            path, hash_value, hash_algorithm=hash_algorithm, hash_kwargs=hash_kwargs
        )


def import_from_url(
    url: str, fname: Optional[PathLike] = None, *args: Any, **kwargs: Any
) -> None:
    """Wrap download so that it works with Kaggle and GitHub."""
    parsed_url = urlparse(url)
    path = Path(url.split("/")[-1] if fname is None else fname)
    logger = logging.getLogger(__name__)
    logger.info("Downloading %s to %s", url, path)

    if parsed_url.netloc == "www.kaggle.com":
        kaggle_name = Path(parsed_url.path).name
        subprocess.run(
            f"kaggle competitions download -p {path!s} -c {kaggle_name}",
            shell=True,
            check=True,
        )
    elif parsed_url.netloc == "github.com":
        subprocess.run(
            f"svn export --force {url.replace('tree/main', 'trunk')} {path!s}",
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
#                 f"Ignore this warning if the file format is parquet."
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
#                 f"Ignore this warning if the file format is parquet."
#             )
#         elif file.name in reference and filehash != reference[file.name]:
#             warnings.warn(
#                 f"File {file.name!r} failed to validate!"
#                 f"File hash {filehash!r} does not match reference {reference[file.name]!r}."
#                 f"Ignore this warning if the file format is parquet."
#             )
#         elif file.stem in reference and filehash != reference[file.stem]:
#             warnings.warn(
#                 f"File {file.name!r} failed to validate!"
#                 f"File hash {filehash!r} does not match reference {reference[file.stem]!r}."
#                 f"Ignore this warning if the file format is parquet."
#             )
#         else:
#             self.LOGGER.info(
#                 f"File {file.name!r} validated successfully {filehash=!r}."
#             )
#     else:
#         raise TypeError(f"Unsupported type for {reference=}.")
#
#     self.LOGGER.debug("Finished validating file.")
