# python
r"""Implements a downloader for the TSDM-package (httpx-based)."""

__all__ = [
    # Functions
    "download",
    "download_directory_to_zip",
    "download_from_github",
    "download_from_kaggle",
    "import_from_url",
]

import logging
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, nullcontext
from html.parser import HTMLParser
from io import IOBase
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import IO, Any, Optional
from urllib.parse import unquote, urljoin, urlparse
from zipfile import ZIP_DEFLATED, ZipFile

from httpx2 import Client, Response
from tqdm.auto import tqdm

from tsdm.constants import EMPTY_MAP
from tsdm.testing.validation import validate_file_hash
from tsdm.types.aliases import FilePath

from .contextmanagers import timer

_DEFAULT_CHUNK_SIZE = 1024 * 1024
r"""Default chunk size for downloads (1 MiB)."""
_DEFAULT_TIMEOUT = 10
r"""Default timeout for requests (connect, read)."""


def download_directory_to_zip(
    url: str,
    zip_filename: FilePath,
    *,
    # zip options
    add_toplevel_dir: bool = False,
    zip_options: Mapping[str, Any] = EMPTY_MAP,
    # concurrency options
    max_workers: int = 8,
    # client options
    username: Optional[str] = None,
    password: Optional[str] = None,
    headers: Mapping[str, str] = EMPTY_MAP,
    timeout: Optional[float] = _DEFAULT_TIMEOUT,
    # auxiliary request options
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
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
    with (
        _delete_if_failed(zipfile_path),
        ZipFile(zipfile_path, **zip_options) as archive,  # pyrefly: ignore[no-matching-overload]
        ThreadPoolExecutor(max_workers=max_workers) as pool,
        Client(auth=auth, headers=headers, timeout=timeout) as client,
    ):
        response = client.get(url)
        response.raise_for_status()

        if not (content := sorted(_yield_suburls(url, client=client))):
            raise RuntimeError(f"No files found at {url}.")

        filenames = {
            href: stem / _sanitize_zip_path(_url_to_relative_path(url, href))
            for href in content
        }

        futures = {
            pool.submit(
                _fetch_bytes,
                client,
                href,
                chunk_size=chunk_size,
                **request_options,
            ): href
            for href in content
        }

        for fut in tqdm(as_completed(futures), total=len(futures)):
            href = futures[fut]
            data = fut.result()
            with archive.open(str(filenames[href]), "w", force_zip64=True) as f:
                f.write(data)


def download_from_kaggle(
    url: str,
    fname: FilePath,
    /,
    timeout: Optional[float] = _DEFAULT_TIMEOUT,
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
    with timer(timeout=timeout):
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
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    # file options
    skip_existing: bool = False,
    hash_value: Optional[str] = None,
    hash_algorithm: Optional[str] = None,
) -> None:
    r"""Download a file from a URL.

    This is essentially a wrapper around `httpx` with a progress bar.
    """
    if isinstance(fname, IOBase | IO):
        _download_io(
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
    with (
        _delete_if_failed(path),
        path.open("wb") as file,
    ):
        _download_io(
            url,
            file,
            username=username,
            password=password,
            headers=headers,
            request_options=request_options,
            chunk_size=chunk_size,
        )

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


def _download_io(
    url: str,
    file: IO[bytes],
    *,
    client: Optional[Client] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    # auxiliary request options
    headers: Mapping[str, str] = EMPTY_MAP,
    timeout: Optional[float] = _DEFAULT_TIMEOUT,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    request_options: Mapping[str, Any] = EMPTY_MAP,
) -> None:
    r"""Download a file from a URL to an IO stream using httpx."""
    request_opts: Mapping[str, Any] = {
        "headers": headers,
        "auth": None if username is None else (username, password),
        "timeout": timeout,
    } | dict(request_options)
    with (
        (
            Client(headers=headers, timeout=timeout)
            if client is None
            else nullcontext(client)
        ) as c,
        c.stream("GET", url, **request_opts) as response,
    ):
        response.raise_for_status()
        with tqdm(
            desc=f"Downloading {url}",
            total=_get_content_length(response),
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as progress_bar:
            for data in response.iter_bytes(chunk_size=chunk_size):
                if data:
                    file.write(data)
                    progress_bar.update(len(data))
                    progress_bar.refresh()


class _delete_if_failed(AbstractContextManager):
    r"""Context manager for downloading a file from a URL to an IO stream using httpx."""

    def __init__(self, path: Path, /) -> None:
        self.path = path
        if self.path.exists():
            raise FileExistsError(f"File {self.path} already exists!")

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> None:
        if not self.path.exists():
            raise RuntimeError(f"File {self.path} does not exist on exit!")
        if exc_value is not None:
            exc_value.add_note(
                f"Exception occurred while downloading {self.path},"
                f" deleting partially downloaded file."
            )
            self.path.unlink()


def _get_content_length(response: Response, /) -> Optional[int]:
    r"""Get the content length in bytes from an httpx Response."""
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    return int(content_length)


def _url_to_relative_path(base_url: str, file_url: str) -> PurePosixPath:
    r"""Convert a URL to a relative path based on a base URL.

    Robost alternative to `os.path.relpath(base_url, url)`.
    """
    base = urlparse(base_url)
    file = urlparse(file_url)

    if (base.scheme != file.scheme) or (base.netloc != file.netloc):
        raise ValueError(f"URL {file_url!r} is not under base URL {base_url!r}.")

    base_path = PurePosixPath(unquote(base.path))
    file_path = PurePosixPath(unquote(file.path))
    return file_path.relative_to(base_path)


class _LinkParser(HTMLParser):
    r"""Parse links from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    self.links.append(value)


def _yield_suburls(url: str, /, *, client: Client) -> Iterator[str]:
    r"""Yield recursively suburls from an url using httpx."""
    response = client.get(url)
    response.raise_for_status()

    parser = _LinkParser()
    parser.feed(response.text)

    for link in parser.links:
        if link == "../":
            continue
        if link.endswith("/"):  # Recursion
            yield from _yield_suburls(urljoin(url, link), client=client)
        else:
            yield urljoin(url, link)


def _fetch_bytes(
    client: Client,
    url: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    **request_options: Any,
) -> bytes:
    r"""Download a URL and return its content as bytes using httpx."""
    with client.stream("GET", url, **request_options) as response:
        response.raise_for_status()
        return b"".join(response.iter_bytes(chunk_size=chunk_size))


def _sanitize_zip_path(filepath: FilePath, /) -> PurePosixPath:
    r"""Sanitize a file path for inclusion in a zip archive."""
    path = PurePosixPath(filepath)

    if "\x00" in str(path):
        raise ValueError(f"Null byte in zip path: {path!r}")
    if "\\" in str(path):
        raise ValueError(f"Backslashes are not allowed in zip paths: {path!r}")
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in zip files: {path!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe path segments in zip path: {path!r}")
    return path
