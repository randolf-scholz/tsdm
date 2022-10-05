r"""Base Class for pretrained models."""

__all__ = [
    # Classes
    "PreTrainedMetaClass",
    "PreTrainedModel",
]

import inspect
import logging
import os
import subprocess
import warnings
import webbrowser
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import torch

from tsdm.config import MODELDIR
from tsdm.utils import paths_exists
from tsdm.utils.remote import download
from tsdm.utils.types import Nested


class PreTrainedMetaClass(ABCMeta):
    r"""Metaclass for `PreTrainedModel`."""

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attributes: dict[str, Any],
        **kwds: dict[str, Any],
    ):
        super().__init__(name, bases, attributes, **kwds)

        if "LOGGER" not in attributes:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if "RAWDATA_DIR" not in attributes:
            if os.environ.get("GENERATING_DOCS", False):
                cls.MODELDIR = Path(f"~/.tsdm/models/{cls.__name__}/")
            else:
                cls.MODELDIR = MODELDIR / cls.__name__


class PreTrainedModel(ABC, torch.nn.Module, metaclass=PreTrainedMetaClass):
    r"""Base class for all pretrained models."""

    LOGGER: logging.Logger
    MODELDIR: Path
    INFO_URL: Optional[str] = None

    def __new__(  # type: ignore[misc]
        cls, *, initialize: bool = True, reset: bool = False
    ) -> torch.nn.Module:
        r"""Initialize the dataset."""
        self: PreTrainedModel = super().__new__(cls)
        if not inspect.isabstract(self):
            self.MODELDIR.mkdir(parents=True, exist_ok=True)
        if reset:
            self.download()
        if initialize:
            return self.load()
        return self

    def load(self) -> torch.nn.Module:
        r"""Load the module."""
        model = torch.jit.load(self.model_path)
        return model

    @classmethod
    def info(cls):
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    @abstractmethod
    @property
    def model_file(self) -> str:
        r"""Add url where to download the dataset from."""

    @cached_property
    def model_path(self) -> Path:
        r"""Return the model path."""
        return self.MODELDIR / self.model_file

    @abstractmethod
    @property
    def download_url(self) -> str:
        r"""Add url where to download the dataset from."""

    def download(self) -> None:
        r"""Download model from url."""
        self.download_from_url(self.download_url)

    @classmethod
    def download_from_url(cls, url: str) -> None:
        r"""Download files from a URL."""
        cls.LOGGER.info("Downloading from %s", url)
        parsed_url = urlparse(url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                f"kaggle competitions download -p {cls.MODELDIR} -c {kaggle_name}",
                shell=True,
                check=True,
            )
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                f"svn export --force {url.replace('tree/main', 'trunk')} {cls.MODELDIR}",
                shell=True,
                check=True,
            )
        else:  # default parsing, including for UCI dataset
            fname = url.split("/")[-1]
            download(url, cls.MODELDIR / fname)

    def validate(
        self,
        filespec: Nested[str | Path],
        /,
        *,
        reference: Optional[str | Mapping[str, str]] = None,
    ) -> None:
        r"""Validate the file hash."""
        self.LOGGER.debug("Starting to validate dataset")

        if isinstance(filespec, Mapping):
            for value in filespec.values():
                self.validate(value, reference=reference)
            return
        if isinstance(filespec, Sequence) and not isinstance(filespec, (str, Path)):
            for value in filespec:
                self.validate(value, reference=reference)
            return

        assert isinstance(filespec, (str, Path)), f"{filespec=} wrong type!"
        file = Path(filespec)

        if not file.exists():
            raise FileNotFoundError(f"File '{file.name}' does not exist!")

        filehash = sha256(file.read_bytes()).hexdigest()

        if reference is None:
            warnings.warn(
                f"File '{file.name}' cannot be validated as no hash is stored in {self.__class__}."
                f"The filehash is '{filehash}'."
            )

        elif isinstance(reference, str):
            if filehash != reference:
                warnings.warn(
                    f"File '{file.name}' failed to validate!"
                    f"File hash '{filehash}' does not match reference '{reference}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            self.LOGGER.info(
                f"File '{file.name}' validated successfully '{filehash=}'."
            )

        elif isinstance(reference, Mapping):
            if not (file.name in reference) ^ (file.stem in reference):
                warnings.warn(
                    f"File '{file.name}' cannot be validated as it is not contained in {reference}."
                    f"The filehash is '{filehash}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            elif file.name in reference and filehash != reference[file.name]:
                warnings.warn(
                    f"File '{file.name}' failed to validate!"
                    f"File hash '{filehash}' does not match reference '{reference[file.name]}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            elif file.stem in reference and filehash != reference[file.stem]:
                warnings.warn(
                    f"File '{file.name}' failed to validate!"
                    f"File hash '{filehash}' does not match reference '{reference[file.stem]}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            else:
                self.LOGGER.info(
                    f"File '{file.name}' validated successfully '{filehash=}'."
                )
        else:
            raise TypeError(f"Unsupported type for {reference=}.")

        self.LOGGER.debug("Finished validating file.")

    def model_files_exist(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.model_path)
