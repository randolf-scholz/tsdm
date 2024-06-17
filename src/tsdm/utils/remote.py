r"""Implements a downloader for the TSDM-package."""

__all__ = [
    # Classes
    "LinkParser",
    # Functions
    "download",
    "download_directory_to_zip",
    "download_from_github",
    "download_from_kaggle",
    "download_io",
    "import_from_url",
    "iter_content",
    "stream_download",
]

import logging
import os
import shutil
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
from tsdm.types.aliases import FilePath


class LinkParser(HTMLParser):
    r"""Parse links from an HTML page."""

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
    r"""Download a file from a URL to an IO stream."""
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
    r"""Download a file as a bytes-stream."""
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
    r"""Iterate over the contents of a directory."""
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
#
# - "headers"
# - "cookies"
# - "auth"
# - "proxies"
# - "hooks"
# - "params"
# - "verify"
# - "cert"
# - "adapters"
# - "stream"
# - "trust_env"
# - "max_redirects"
#
def download_directory_to_zip(
    url: str,
    zip_filename: FilePath,
    *,
    # session options
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Mapping[str, str] = EMPTY_MAP,
    stream: bool = True,
    add_toplevel_dir: bool = True,
    zip_options: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    r"""Download a directory from a URL to a zip file."""
    path = Path(zip_filename)
    stem = f"{path.stem}/" if add_toplevel_dir else ""
    zip_options = {"mode": "w"} | dict(zip_options)

    # validate the path
    if path.suffix != ".zip":
        raise ValueError(f"{path=} must have .zip suffix!")

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

                with archive.open(stem + file_name, "w", force_zip64=True) as file:
                    download_io(href, file, session=session)


def download_from_kaggle(url: str, fname: FilePath, /, **options: Any) -> None:
    r"""Import a dataset from Kaggle."""
    # check that kaggle is installed
    if not shutil.which("kaggle"):
        raise RuntimeError("Kaggle CLI is not installed!")

    # check that the URL is a Kaggle URL
    parsed_url = urlparse(url)
    if not parsed_url.netloc == "www.kaggle.com":
        raise ValueError(f"{url=} is not a Kaggle URL!")

    # construct the path
    path = Path(fname)
    target_directory = path.parent
    if not target_directory.exists():
        target_directory.mkdir(parents=True, exist_ok=True)

    # download the dataset
    kaggle_name = Path(urlparse(url).path).name
    kaggle_opts = " ".join(f"--{k} {v}" for k, v in options.items())
    subprocess.run(
        "kaggle competitions download"
        f" -p {target_directory} -c {kaggle_name} {kaggle_opts}",
        shell=True,
        check=True,
    )


def download_from_github(url: str, fname: FilePath, /, **options: Any) -> None:
    r"""Import a file from GitHub."""
    # check that svn is installed
    if not shutil.which("svn"):
        raise RuntimeError("Subversion (svn) is not installed!")

    # check that the URL is a GitHub URL
    parsed_url = urlparse(url)
    if not parsed_url.netloc == "github.com":
        raise ValueError(f"{url=} is not a GitHub URL!")

    # construct the path
    path = Path(fname)
    target_directory = path.parent
    if not target_directory.exists():
        target_directory.mkdir(parents=True, exist_ok=True)

    # download the file
    svn_url = url.replace("tree/main", "trunk")
    svn_opts = " ".join(f"--{k} {v}" for k, v in options.items())
    subprocess.run(
        f"svn export --force {svn_url} {target_directory} {svn_opts}",
        shell=True,
        check=True,
    )


def download(
    url: str,
    fname: Optional[FilePath | IOBase] = None,
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
        return download_io(
            url,
            fname,
            username=username,
            password=password,
            headers=headers,
            request_options=request_options,
            chunk_size=chunk_size,
        )

    # construct the path
    path = Path(url.split("/")[-1] if fname is None else fname)
    target_directory = path.parent
    if not target_directory.exists():
        target_directory.mkdir(parents=True, exist_ok=True)

    # check if the file already exists
    if skip_existing and path.exists():
        # skip download, but validate the hash
        if hash_value is not None:
            validate_file_hash(
                path,
                hash_value,
                hash_alg=hash_algorithm,
                hash_kwargs=hash_kwargs,
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
        exc.add_note(f"Error occurred while downloading {fname}, deleting files.")
        raise

    # validate the file hash
    if hash_value is not None:
        validate_file_hash(
            path,
            hash_value,
            hash_alg=hash_algorithm,
            hash_kwargs=hash_kwargs,
        )


def import_from_url(
    url: str, fname: Optional[FilePath] = None, *args: Any, **kwargs: Any
) -> None:
    r"""Wrap download so that it works with Kaggle and GitHub."""
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
