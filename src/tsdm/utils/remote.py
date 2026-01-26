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
    "yield_suburls",
]

import logging
import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from io import IOBase
from pathlib import Path
from typing import IO, Any, Optional
from urllib.parse import urljoin, urlparse
from zipfile import ZIP_DEFLATED, ZipFile

import requests
from requests import Session
from tqdm.auto import tqdm

from tsdm.config import CONFIG
from tsdm.constants import EMPTY_MAP
from tsdm.testing.validation import validate_file_hash
from tsdm.types.aliases import FilePath

DEFAULT_CHUNK_SIZE = 1024 * 1024
r"""Default chunk size for downloads (1 MiB)."""
DEFAULT_TIMEOUT = 10
r"""Default timeout for requests (connect, read)."""

assert DEFAULT_CHUNK_SIZE == CONFIG.DEFAULT_CHUNK_SIZE

# NOTE: Session options as of requests 2.26.0
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


class LinkParser(HTMLParser):
    r"""Parse links from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    self.links.append(value)


def download_io(
    url: str,
    file: IO[bytes],
    *,
    session: Optional[Session] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    # auxiliary request options
    headers: Mapping[str, str] = EMPTY_MAP,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    request_options: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    r"""Download a file from a URL to an IO stream."""
    options: dict[str, Any] = {
        "headers": headers,
        "auth": None if username is None else (username, password),
        "stream": True,
        "timeout": timeout,
    } | dict(request_options)

    with (
        requests.get(url, **options)  # noqa: S113
        if session is None
        else session.get(url, **options)
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0) or 0) or None
        with tqdm(
            desc=f"Downloading {url}",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=chunk_size):
                if data:
                    file.write(data)
                    progress_bar.update(len(data))


def _fetch_bytes(
    session: Session, url: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> bytes:
    r"""Download a URL and return its content as bytes."""
    with session.get(url, stream=True) as response:
        response.raise_for_status()
        return b"".join(response.iter_content(chunk_size=chunk_size))


def yield_suburls(url: str, /, *, session: Session) -> Iterator[str]:
    r"""Yield recursively suburls from an url."""
    with session.get(url) as response:
        response.raise_for_status()

        parser = LinkParser()
        parser.feed(response.text)

        for link in parser.links:
            if link == "../":
                continue
            if link.endswith("/"):  # Recursion
                yield from yield_suburls(urljoin(url, link), session=session)
            else:
                yield url + link


def download_directory_to_zip(
    url: str,
    zip_filename: FilePath,
    *,
    # zip options
    add_toplevel_dir: bool = True,
    zip_options: Mapping[str, Any] = EMPTY_MAP,
    # concurrency options
    max_workers: int = 8,
    # session options
    username: Optional[str] = None,
    password: Optional[str] = None,
    stream: bool = True,
    # auxiliary request options
    headers: Mapping[str, str] = EMPTY_MAP,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    request_options: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    r"""Download a directory from a URL to a zip file (concurrent fetch, serial zip write)."""
    zipfile_path = Path(zip_filename)
    stem = f"{zipfile_path.stem}/" if add_toplevel_dir else ""
    zip_options = {"mode": "w", "compression": ZIP_DEFLATED} | dict(zip_options)

    if zipfile_path.suffix != ".zip":
        raise ValueError(f"{zipfile_path=} must have .zip suffix!")

    auth = (
        (username, password) if username is not None and password is not None else None
    )

    with Session() as session:
        session.headers.update(headers)
        session.stream = stream
        session.auth = auth

        with session.get(url) as response:
            response.raise_for_status()

            if not (content := sorted(yield_suburls(url, session=session))):
                raise RuntimeError(f"No files found at {url}.")

            # Fetch in parallel, write sequentially to avoid corrupting the zip.
            with (
                ZipFile(zipfile_path, **zip_options) as archive,  # type: ignore[call-overload]
                ThreadPoolExecutor(max_workers=max_workers) as pool,
                tqdm(total=len(content), leave=False) as pbar,
            ):
                futures = {
                    pool.submit(
                        _fetch_bytes, session, href, chunk_size=chunk_size
                    ): href
                    for href in content
                }

                for fut in as_completed(futures, timeout=timeout):
                    href = futures[fut]
                    file_name = os.path.relpath(href, url)
                    if ".." in file_name:
                        raise ValueError(f"File name cannot contain '..': {file_name=}")

                    pbar.set_description(f"Downloading {file_name}")

                    data = fut.result()
                    with archive.open(stem + file_name, "w", force_zip64=True) as f:
                        f.write(data)

                    pbar.update(1)


def download_from_kaggle(
    url: str,
    fname: FilePath,
    /,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    **kaggle_options: Any,
) -> None:
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
    kaggle_opts = [
        item for k, v in kaggle_options.items() for item in (f"--{k}", str(v))
    ]
    subprocess.run(
        [  # noqa: S607
            "kaggle",
            "competitions",
            "download",
            "-p",
            str(target_directory),
            "-c",
            kaggle_name,
            *kaggle_opts,
        ],
        check=True,
    )


def download_from_github(url: str, fname: FilePath, /, **svn_options: Any) -> None:
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
    svn_opts = [item for k, v in svn_options.items() for item in (f"--{k}", str(v))]
    subprocess.run(
        [
            "/usr/bin/svn",
            "export",
            "--force",
            svn_url,
            str(target_directory),
            *svn_opts,
        ],
        check=True,
    )


def download(
    url: str,
    fname: Optional[FilePath | IO[bytes]] = None,
    *,
    # request options
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Mapping[str, str] = EMPTY_MAP,
    request_options: Mapping[str, Any] = EMPTY_MAP,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    # file options
    skip_existing: bool = False,
    hash_value: Optional[str] = None,
    hash_algorithm: Optional[str] = None,
) -> None:
    r"""Download a file from a URL.

    This is essentially a wrapper around `requests.get` with a progress bar.
    """
    if isinstance(fname, IOBase | IO):
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
    path = Path(url.rsplit("/", maxsplit=1)[-1] if fname is None else fname)
    target_directory = path.parent
    if not target_directory.exists():
        target_directory.mkdir(parents=True, exist_ok=True)

    # check if the file already exists
    if skip_existing and path.exists():
        # skip download, but validate the hash
        if hash_value is not None:
            validate_file_hash(path, hash_value, hash_algorithm=hash_algorithm)
        return

    # attempt to download the file
    try:
        with path.open("wb") as file:
            download_io(
                url,
                file,
                username=username,
                password=password,
                headers=headers,
                request_options=request_options,
                chunk_size=chunk_size,
            )
    except BaseException as exc:
        path.unlink()
        exc.add_note(
            f"Exception occurred while downloading {fname}, deleting partially downloaded file."
        )
        raise

    # validate the file hash
    if hash_value is not None:
        validate_file_hash(path, hash_value, hash_algorithm=hash_algorithm)


def import_from_url(
    url: str, fname: Optional[FilePath] = None, /, *args: Any, **kwargs: Any
) -> None:
    r"""Wrap download so that it works with Kaggle and GitHub."""
    parsed_url = urlparse(url)
    path = Path(url.rsplit("/", maxsplit=1)[-1] if fname is None else fname)
    logger = logging.getLogger(__name__)
    logger.info("Downloading %s to %s", url, path)

    match parsed_url.netloc:
        case "www.kaggle.com":
            download_from_kaggle(url, path, **kwargs)
        case "github.com":
            download_from_github(url, path, **kwargs)
        case _:  # default parsing, including for UCI dataset
            download(url, path, *args, **kwargs)
