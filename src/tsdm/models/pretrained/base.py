r"""Base Class for pretrained models.

A Pretrained Model is a model that has been trained on a specific task.
It can be used to make predictions without having to train the model again.

Therefore, a pretrained model is a combination of a model, a task and an encoder/decoder.

We allow several ways of initializing Pretrained Models:

1. Load a pretrained model from a local directory or file
    - keep the file in its original location and load it from there
2. Load a pretrained model from a remote directory or file
    - download the file to a local directory and load it from there
3. Load a pretrained model from a remote database via a key / identifier
    - download the file to a local directory and load it from there


A model checkpoint might consist of several files, such as:

- The model weights
- The optimizer state
- The training history
- The hyperparameters

Therefore, we need to define a way of storing and loading these files.

- There should be some way to automatically determine which components are available
"""

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
import warnings
import webbrowser
from abc import ABC, ABCMeta
from collections.abc import Collection, Mapping
from functools import cached_property
from io import IOBase
from pathlib import Path
from typing import IO, Any, ClassVar, Optional, cast
from zipfile import ZipFile

import torch
import yaml
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler

from tsdm.config import CONFIG
from tsdm.encoders import BaseEncoder
from tsdm.optimizers import LR_SCHEDULERS, OPTIMIZERS
from tsdm.types.aliases import PathLike
from tsdm.utils import (
    LazyDict,
    initialize_from_config,
    is_zipfile,
    paths_exists,
    repackage_zip,
)
from tsdm.utils.remote import import_from_url
from tsdm.utils.strings import repr_mapping

# TODO: loading components!


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
                cls.RAWDATA_DIR = CONFIG.MODELDIR / cls.__name__


class PreTrainedModel(ABC, metaclass=PreTrainedMetaClass):
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
    """Logger for the class."""
    CHECKPOINT_URL: ClassVar[Optional[str]] = None
    """URL with overview of available model checkpoints."""
    DOWNLOAD_URL: ClassVar[Optional[str]] = None
    """URL from which model checkpoints can be downloaded."""
    DOCUMENTATION_URL: ClassVar[Optional[str]] = None
    """URL of online documentation for the model."""
    RAWDATA_DIR: ClassVar[Path]
    """Default directory where the raw data is stored."""

    # Instance Variables
    components: Mapping[str, Any]
    """Mapping of component names to their respective components."""

    device: str | torch.device
    """Device which the model components are loaded to."""
    rawdata_path: Path = NotImplemented
    """Path where the raw data is stored (subpath of RAWDATA_DIR)."""
    rawdata_hash: Optional[dict[str, str]] = None
    """Dictionary of hash-method:value pairs for the raw data."""
    download_url: Optional[str] = None
    """URL from which the raw data can be downloaded."""

    __component_files: dict[str, PathLike]

    __component_aliases: dict[str, Collection[str]] = {
        "optimizer": OPTIMIZERS,
        "lr_scheduler": LR_SCHEDULERS,
        "model": ["model", "Model", "LinODEnet"],
        "encoder": ["encoder", "Encoder"],
        "hyperparameters": ["hyperparameters", "hparams", "hyperparameter", "hparam"],
    }
    """Mapping of potential component aliases for the model."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        initialize: bool = True,
        reset: bool = False,
        rawdata_path: Optional[Path] = None,
        rawdata_hash: Optional[dict[str, str]] = None,
        download_url: Optional[str] = None,
    ) -> None:
        if rawdata_path is not None:
            self.rawdata_path = Path(rawdata_path)
        if download_url is not None:
            self.download_url = download_url
        if rawdata_hash is not None:
            self.rawdata_hash = rawdata_hash

        if self.rawdata_path is NotImplemented:
            raise NotImplementedError(
                f"\n{self.__class__.__name__} does not implement a default model!"
                f"\nPlease use one of the constructor methods to specify the model."
                f"\nAvailable models are listed in {self.available_checkpoints}."
            )

        if not inspect.isabstract(self):
            if reset:
                self.rawdata_path.rmdir()
            if self.rawdata_path.is_dir():
                self.rawdata_path.mkdir(parents=True, exist_ok=True)
            if initialize and self.download_url is not None:
                self.download()
                repackage_zip(self.rawdata_path)

        self.component_files = self.autodetect_component_files()

        self.components = LazyDict(
            {k: self.get_component for k in self.component_files}
        )
        self.device = device

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return repr_mapping(self.components, wrapped=self, identifier="PreTrainedModel")

    @classmethod
    def available_checkpoints(cls) -> None:
        r"""Return a dictionary of available checkpoints."""
        if cls.CHECKPOINT_URL is None:
            raise NotImplementedError(
                f"{cls.__name__} does not implement a checkpoint documentation URL!"
            )
        webbrowser.open_new_tab(cls.CHECKPOINT_URL)

    @classmethod
    def info(cls) -> None:
        r"""Open information in about the model."""
        if cls.DOCUMENTATION_URL is None:
            raise NotImplementedError(
                f"{cls.__name__} does not implement a model documentation URL!"
            )
        webbrowser.open_new_tab(cls.DOCUMENTATION_URL)

    @classmethod
    def from_path(
        cls, file_or_directory: str | Path, /, *args: Any, **kwargs: Any
    ) -> Any:
        r"""Create model from directory."""
        path = Path(file_or_directory)
        if not path.is_relative_to(Path.cwd()):
            path = Path.cwd() / path
        if not path.exists():
            raise ValueError(f"{path} does not exist!")
        return cls(*args, rawdata_path=path, **kwargs)

    @classmethod
    def from_url(
        cls, url: str, /, *args: Any, **kwargs: Any
    ) -> Any:  # FIXME: Return Self PreTrainedModel
        r"""Obtain model from arbitrary url."""
        fname = url.split("/")[-1]
        path = cls.RAWDATA_DIR / fname
        return cls(*args, rawdata_path=path, download_url=url, **kwargs)

    @classmethod
    def from_remote_checkpoint(
        cls, checkpoint: str, /, *args: Any, **kwargs: Any
    ) -> Any:  # FIXME: Use typing.Self
        r"""Create model from local."""
        if cls.DOWNLOAD_URL is None:
            raise NotImplementedError(
                f"{cls.__name__} does not implement a checkpoint url!"
            )
        url = cls.DOWNLOAD_URL + checkpoint
        path = cls.RAWDATA_DIR / checkpoint
        return cls(*args, rawdata_path=path, download_url=url, **kwargs)

    @cached_property
    def model(self) -> torch.nn.Module:
        r"""Return the model."""
        return self.components.get("model", NotImplemented)

    @cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        r"""Return the optimizer."""
        return self.components.get("optimizer", NotImplemented)

    @cached_property
    def hyperparameters(self) -> dict[str, Any]:
        r"""Return the hyperparameters."""
        return self.components.get("hyperparameters", NotImplemented)

    @cached_property
    def encoder(self, /) -> BaseEncoder:
        r"""Return the encoder."""
        return self.components.get("encoder", NotImplemented)

    @cached_property
    def lr_scheduler(self) -> TorchLRScheduler:
        r"""Return the learning rate scheduler."""
        return self.components.get("lr_scheduler", NotImplemented)

    def autodetect_component_files(self) -> dict[str, str]:
        r"""Detect which components are available."""
        if is_zipfile(self.rawdata_path):
            with ZipFile(self.rawdata_path, "r") as zf:
                return {Path(f).stem: f for f in zf.namelist()}
        if self.rawdata_path.is_dir():
            return {f.stem: f.name for f in self.rawdata_path.iterdir()}
        raise ValueError(f"{self.rawdata_path} unsupported filetype!")

    def detect_components(self) -> list[str]:
        r"""Detect which components are available."""
        if is_zipfile(self.rawdata_path):
            with ZipFile(self.rawdata_path, "r") as zf:
                return [Path(f).stem for f in zf.namelist()]
        if self.rawdata_path.is_dir():
            return [f.stem for f in self.rawdata_path.iterdir()]
        raise ValueError(f"{self.rawdata_path} unsupported filetype!")

    def download(self, *, force: bool = False) -> None:
        r"""Download model from url."""
        if self.rawdata_path_exists() and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        if self.download_url is None:
            raise NotImplementedError(f"{self} has no url")
        self.LOGGER.debug("Starting to download dataset from %s.", self.download_url)
        import_from_url(self.download_url, self.rawdata_path)
        self.LOGGER.debug("Finished downloading dataset from %s.", self.download_url)

    def rawdata_path_exists(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.rawdata_path)

    def get_component(self, component: str, /) -> Any:
        r"""Extract a model component."""
        if not self.rawdata_path_exists():
            self.LOGGER.debug("Obtaining rawdata file!")
            self.download()
        if component not in self.components:
            raise ValueError(f"{component=} is not a valid component!")

        file = self.component_files[component]
        extension = Path(file).suffix

        # if directory
        if self.rawdata_path.is_dir():
            file_path = self.rawdata_path / file
            if file_path.suffix in {".yaml", ".json"}:
                with open(file_path, "r", encoding="utf8") as f:
                    return self.__load_component(f, component, extension)
            with open(file_path, "rb") as f:
                return self.__load_component(f, component, extension)

        # if zipfile
        if is_zipfile(self.rawdata_path):
            with ZipFile(self.rawdata_path) as zf:
                with zf.open(file) as f:
                    return self.__load_component(f, component, extension)

        raise ValueError(f"{self.rawdata_path=} is not a zipfile or directory!")

    def __load_component(self, file: IO, component: str, extension: str) -> Any:
        match component, extension:
            case "model", _:
                return self.__load_torch_model(file)
            case "optimizer", _:
                return self.__load_torch_optimizer(file)
            case "lr_scheduler", _:
                return self.__load_torch_lr_scheduler(file)
            case _, ".json":
                return json.load(file)
            case _, (".yaml" | ".yml"):
                return yaml.safe_load(file)
            case _, ".pickle":
                return pickle.load(file)
            case _, _:
                return self.__load_torch_component(component, file)
        raise ValueError(f"{component=} is not supported!")

    def __load_torch_component(
        self, component: str, /, file: str | Path | IO[bytes], **kwargs: Any
    ) -> TorchModule:
        r"""Load a torch component."""
        logger = self.LOGGER.getChild(component)

        try:  # attempt loading via torch.jit.load
            logger.info("Trying to load as TorchScript.")
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            return self.__load_torch_jit_model(file)
        except RuntimeError:
            logger.info("Could not load as TorchScript.")

        try:  # attempt loading via torch.load
            logger.info("Loading as regular torch component.")
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            loaded_object = torch.load(file, map_location=self.device)
        except RuntimeError:
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            loaded_object = torch.load(file, map_location=torch.device("cpu"))
            warnings.warn("Could not load to default device. Loaded to CPU.")

        if not isinstance(loaded_object, dict):
            return loaded_object
        logger.info("Found state_dict.")
        state_dict = loaded_object

        try:
            logger.info("Trying to obtain module config from hyperparameters.")
            module_config = self.hyperparameters[component]
        except KeyError as exc:
            raise ValueError(
                f"No {component!r} configuration in {self.hyperparameters}."
            ) from exc

        module = initialize_from_config(module_config | kwargs)
        module.load_state_dict(state_dict)
        return module

    def __load_torch_jit_model(self, file: str | Path | IO[bytes], /) -> TorchModule:
        r"""Load a TorchScript model."""
        try:
            # load on CPU
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            model = torch.jit.load(file, map_location=self.device)
        except RuntimeError:
            if isinstance(file, IOBase):  # type: ignore[unreachable]
                file.seek(0, 0)
            model = torch.jit.load(file, map_location=torch.device("cpu"))
            warnings.warn("Could not load model to default device. Loaded to CPU.")
        return model

    def __load_torch_model(self, file: str | Path | IO[bytes], /) -> TorchModule:
        r"""Load a torch model."""
        return self.__load_torch_component("model", file)

    def __load_torch_optimizer(self, file: str | Path | IO[bytes], /) -> TorchOptimizer:
        r"""Load a torch optimizer."""
        return cast(
            TorchOptimizer,
            self.__load_torch_component(
                "optimizer", file, params=self.model.parameters()
            ),
        )

    def __load_torch_lr_scheduler(
        self,
        file: str | Path | IO[bytes],
        /,  # FIXME: https://github.com/PyCQA/pycodestyle/issues/951
    ) -> TorchLRScheduler:
        r"""Load a torch learning rate scheduler."""
        return cast(TorchLRScheduler, self.__load_torch_component("lr_scheduler", file))

    # def load(self, *, force: bool = False, validate: bool = True) -> torch.nn.Module:
    #     r"""Load the selected DATASET_OBJECT."""
    #     if not self.rawdata_path_exists():
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
