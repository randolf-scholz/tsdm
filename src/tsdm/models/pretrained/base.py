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
from io import IOBase
from pathlib import Path
from typing import IO, Any, ClassVar, Optional
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
        **kwds: Any,
    ) -> None:
        super().__init__(name, bases, attributes, **kwds)

        if "LOGGER" not in attributes:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if "RAWDATA_DIR" not in attributes:
            if os.environ.get("GENERATING_DOCS", False):
                cls.MODELDIR = Path(f"~/.tsdm/models/{cls.__name__}/")
            else:
                cls.MODELDIR = MODELDIR / cls.__name__

    # def __call__(cls, *args: Any, **kw: Any) -> Any:
    #     r"""Prevent __init__."""
    #     return cls.__new__(cls, *args, **kw)


class PreTrainedModel(ABC, torch.nn.Module, metaclass=PreTrainedMetaClass):
    r"""Base class for all pretrained models."""

    # Class Variables
    LOGGER: ClassVar[logging.Logger]
    MODELDIR: Path
    INFO_URL: Optional[str] = None
    MODEL_SHA256: Optional[str] = None
    DOWNLOAD_URL: Optional[str] = None

    # Instance Variables
    device: Optional[str | torch.device] = None

    def __new__(  # type: ignore[misc]
        cls, *, initialize: bool = True, reset: bool = False, **kwds: Any
    ) -> torch.nn.Module:
        r"""Cronstruct the model object and initialize it."""
        self: PreTrainedModel = super().__new__(cls)

        if not inspect.isabstract(self):
            self.MODELDIR.mkdir(parents=True, exist_ok=True)

        self.__init__(**kwds)  # type: ignore[misc]

        if reset:
            self.download()
        if initialize:
            return self.load()
        return self

    def __init__(self, *, device: Optional[str | torch.device] = None) -> None:
        super().__init__()
        self.device = device

    def load(self, *, force: bool = False, validate: bool = True) -> torch.nn.Module:
        r"""Load the selected DATASET_OBJECT."""
        if not self.model_files_exist():
            self.download(force=force, validate=validate)
        else:
            self.LOGGER.debug("Dataset files already exist!")

        if validate:
            self.validate(self.model_path, reference=self.MODEL_SHA256)

        self.LOGGER.debug("Starting to load dataset.")
        ds = self._load()
        self.LOGGER.debug("Finished loading dataset.")
        return ds

    @staticmethod
    def load_torch_jit(
        file: str | Path | IO[bytes],
        /,
        *,
        map_location: Optional[str | torch.device] = None,
        _extra_files: Optional[str | Path | dict] = None,
    ) -> torch.nn.Module:
        r"""Load a TorchScript model from a file."""
        try:
            model = torch.jit.load(
                file, map_location=map_location, _extra_files=_extra_files
            )
        except RuntimeError:
            warnings.warn("Could not load model to default device. Trying CPU.")
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            model = torch.jit.load(
                file, map_location=torch.device("cpu"), _extra_files=_extra_files
            )
        return model

    def _load(self) -> torch.nn.Module:
        r"""Load the module."""
        return self.load_torch_jit(self.model_path, map_location=self.device)

    @classmethod
    def info(cls) -> None:
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    @property
    @abstractmethod
    def model_file(self) -> str:
        r"""Add url where to download the dataset from."""

    @cached_property
    def model_path(self) -> Path:
        r"""Return the model path."""
        return self.MODELDIR / self.model_file

    def download(self, *, force: bool = False, validate: bool = True) -> None:
        r"""Download model from url."""
        if self.model_files_exist() and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        if self.DOWNLOAD_URL is None:
            self.LOGGER.info("Dataset provides no url. Assumed offline")
            assert self.model_files_exist(), "Dataset files do not exist!"

        self.LOGGER.debug("Starting to download dataset.")
        self.download_from_url(self.DOWNLOAD_URL)  # type: ignore[arg-type]
        self.LOGGER.debug("Starting downloading dataset.")

        if validate:
            self.validate(self.model_path, reference=self.MODEL_SHA256)

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