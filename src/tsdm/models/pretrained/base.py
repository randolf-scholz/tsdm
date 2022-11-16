r"""Base Class for pretrained models."""

from __future__ import annotations

__all__ = [
    # Classes
    "PreTrainedMetaClass",
    "PreTrainedModel",
]

import inspect
import json
import logging
import os
import pickle
import subprocess
import warnings
import webbrowser
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property
from hashlib import sha256
from io import IOBase
from pathlib import Path
from typing import IO, Any, ClassVar, Optional, cast
from urllib.parse import urlparse
from zipfile import ZipFile

import torch
import yaml
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler

from tsdm.config import MODELDIR
from tsdm.encoders import BaseEncoder
from tsdm.utils import LazyDict, initialize_from_config, paths_exists
from tsdm.utils.remote import download
from tsdm.utils.strings import repr_mapping, repr_short
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
                cls.RAWDATA_DIR = Path(f"~/.tsdm/models/{cls.__name__}/")
            else:
                cls.RAWDATA_DIR = MODELDIR / cls.__name__

    # def __call__(cls, *args: Any, **kw: Any) -> Any:
    #     r"""Prevent __init__."""
    #     return cls.__new__(cls, *args, **kw)


class PreTrainedModel(Mapping[str, Any], ABC, metaclass=PreTrainedMetaClass):
    r"""Base class for all pretrained models.

    A pretrained model can provide multiple components:
        - the model itself
        - the encoder
        - the optimizer
        - the hyperparameters
        - the learning rate scheduler
    """

    # Class Variables
    LOGGER: ClassVar[logging.Logger]
    # COMPONENT_FILES: ClassVar[dict[str, str]] = NotImplemented
    DOWNLOAD_URL: ClassVar[Optional[str]] = None
    INFO_URL: ClassVar[Optional[str]] = None
    RAWDATA_HASH: ClassVar[Optional[str]] = None
    RAWDATA_DIR: ClassVar[Path]
    IS_JIT: ClassVar[bool] = False

    # Instance Variables
    device: Optional[str | torch.device] = None

    def __new__(
        cls, *, initialize: bool = True, reset: bool = False, **kwds: Any
    ) -> PreTrainedModel:
        r"""Cronstruct the model object and initialize it."""
        self: PreTrainedModel = super().__new__(cls)

        if not inspect.isabstract(self):
            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)

        self.__init__(**kwds)  # type: ignore[misc]

        if reset:
            self.download()
        # if initialize:
        #     return self.load()
        return self

    def __init__(self, *, device: Optional[str | torch.device] = None) -> None:
        super().__init__()
        self.device = device

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, key: str) -> Any:
        return self.components[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __repr__(self) -> str:
        return repr_mapping(self.components, repr_fun=repr_short, wrapped=self)

    @cached_property
    def components(self) -> dict[str, Any]:
        r"""Extract all components."""
        return LazyDict({k: self.get_component for k in self.component_files})

    @property
    @abstractmethod
    def component_files(self) -> Mapping[str, str]:
        r"""Return filename."""

    @property
    @abstractmethod
    def rawdata_file(self) -> str:
        r"""Return filename."""

    @cached_property
    def rawdata_path(self) -> Path:
        r"""Return the rawdata path."""
        return self.RAWDATA_DIR / self.rawdata_file

    def rawdata_file_exist(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.rawdata_path)

    @cached_property
    def model(self) -> torch.nn.Module:
        r"""Return the model."""
        return self.get_component("model")

    @cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        r"""Return the optimizer."""
        return self.get_component("optimizer")

    @cached_property
    def hyperparameters(self) -> dict[str, Any]:
        r"""Return the hyperparameters."""
        return self.get_component("hyperparameters")

    @cached_property
    def encoder(self, /) -> BaseEncoder:
        r"""Return the encoder."""
        return self.get_component("encoder")

    @cached_property
    def lr_scheduler(self) -> TorchLRScheduler:
        r"""Return the learning rate scheduler."""
        return self.get_component("lr_scheduler")

    def get_component(self, component: str, /) -> Any:
        """Extract a model component."""
        if not self.rawdata_file_exist():
            self.LOGGER.debug("Obtaining rawdata file!")
            self.download(validate=True)

        file = self.component_files[component]

        with ZipFile(self.rawdata_path, "r") as archive:
            with archive.open(file, "r") as f:
                match component, Path(file).suffix:
                    case _, ".pickle":
                        return pickle.load(f)
                    case "model", _:
                        return self.__load_torch_model(f)
                    case "optimizer", _:
                        return self.__load_torch_optimizer(f)
                    case "lr_scheduler", _:
                        return self.__load_torch_lr_scheduler(f)
                    case "hyperparameters", ".json":
                        return json.load(f)
                    case "hyperparameters", (".yaml" | ".yml"):
                        return yaml.safe_load(f)
                    case _, _:
                        raise ValueError(f"Unknown format for: {component}")

    def __load_torch_component(
        self, component: str, /, file: str | Path | IO[bytes]
    ) -> TorchModule:
        r"""Load a torch model."""
        try:
            print("Trying to load as a JIT model.")
            return self.__load_torch_jit_model(file)
        except Exception as E:
            warnings.warn(f"{E} was raised. Trying to load as a normal model.")

        try:
            module_config = self.hyperparameters[component]
        except KeyError:
            raise ValueError(
                f"No '{component}' configuration in {self.hyperparameters}."
            )

        module = initialize_from_config(module_config)

        try:
            state_dict = torch.load(file, map_location=self.device)
        except RuntimeError:
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            warnings.warn("Could not load model to default device. Trying CPU.")
            state_dict = torch.load(file, map_location="cpu")

        module.load_state_dict(state_dict)
        return module

    def __load_torch_model(self, file: str | Path | IO[bytes], /) -> TorchModule:
        r"""Load a torch model."""
        try:
            return self.__load_torch_jit_model(file)
        except Exception as E:
            warnings.warn(f"{E} was raised. Trying to load as a normal model.")
            return self.__load_torch_component("model", file)

    def __load_torch_jit_model(self, file: str | Path | IO[bytes], /) -> TorchModule:
        r"""Load a TorchScript model."""
        try:
            model = torch.jit.load(file, map_location=self.device)
        except RuntimeError:
            warnings.warn("Could not load model to default device. Trying CPU.")
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            model = torch.jit.load(file, map_location=torch.device("cpu"))
        return model

    def __load_torch_optimizer(self, file: str | Path | IO[bytes], /) -> TorchOptimizer:
        r"""Load a torch optimizer."""
        return cast(TorchOptimizer, self.__load_torch_component("optimizer", file))

    def __load_torch_lr_scheduler(
        self,
        file: str | Path | IO[bytes],
        /,  # noqa: W504 # FIXME: https://github.com/PyCQA/pycodestyle/issues/951
    ) -> TorchLRScheduler:
        r"""Load a torch learning rate scheduler."""
        return cast(TorchLRScheduler, self.__load_torch_component("lr_scheduler", file))

    # def load(self, *, force: bool = False, validate: bool = True) -> torch.nn.Module:
    #     r"""Load the selected DATASET_OBJECT."""
    #     if not self.rawdata_file_exist():
    #         self.download(force=force, validate=validate)
    #     else:
    #         self.LOGGER.debug("Dataset files already exist!")
    #
    #     if validate:
    #         self.validate(self.rawdata_path, reference=self.RAWDATA_HASH)
    #
    #     self.LOGGER.debug("Starting to load dataset.")
    #     ds = self._load()
    #     self.LOGGER.debug("Finished loading dataset.")
    #     return ds

    @classmethod
    def info(cls) -> None:
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    def download(self, *, force: bool = False, validate: bool = True) -> None:
        r"""Download model from url."""
        if self.rawdata_file_exist() and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        if self.DOWNLOAD_URL is None:
            self.LOGGER.info("Dataset provides no url. Assumed offline")
            assert self.rawdata_file_exist(), "Dataset files do not exist!"

        self.LOGGER.debug("Starting to download dataset.")
        self.download_from_url(self.DOWNLOAD_URL)  # type: ignore[arg-type]
        self.LOGGER.debug("Starting downloading dataset.")

        if validate:
            self.validate(self.rawdata_path, reference=self.RAWDATA_HASH)

    @classmethod
    def download_from_url(cls, url: str) -> None:
        r"""Download files from a URL."""
        cls.LOGGER.info("Downloading from %s", url)
        parsed_url = urlparse(url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                f"kaggle competitions download -p {cls.RAWDATA_DIR} -c {kaggle_name}",
                shell=True,
                check=True,
            )
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                f"svn export --force {url.replace('tree/main', 'trunk')} {cls.RAWDATA_DIR}",
                shell=True,
                check=True,
            )
        else:  # default parsing, including for UCI dataset
            fname = url.split("/")[-1]
            download(url, cls.RAWDATA_DIR / fname)

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
