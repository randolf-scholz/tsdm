r"""Base Class for pretrained models."""

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
from abc import ABC, ABCMeta
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

from tsdm.config import CONFIG
from tsdm.encoders import BaseEncoder
from tsdm.types.aliases import Nested
from tsdm.utils import LazyDict, initialize_from_config, is_zipfile, paths_exists
from tsdm.utils.remote import download
from tsdm.utils.strings import repr_mapping, repr_short


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

    # Instance attributes
    DOWNLOAD_URL: Optional[str] = None
    INFO_URL: Optional[str] = None
    RAWDATA_HASH: Optional[str] = None
    RAWDATA_DIR: Path

    # Instance Variables
    device: Optional[str | torch.device] = None
    components: Mapping[str, Any]
    rawdata_path: Path
    rawdata_file: str = NotImplemented

    component_files: Mapping[str, str] = {
        "model": "model",
        "encoder": "encoder.pickle",
        "optimizer": "optimizer",
        "hyperparameters": "hparams.yaml",
        "lr_scheduler": "lr_scheduler",
    }

    # def __new__(
    #     cls, *, initialize: bool = True, reset: bool = False, **kwds: Any
    # ) -> PreTrainedModel:
    #     r"""Construct the model object and initialize it."""
    #     self: PreTrainedModel = super().__new__(cls)
    #
    #     return self

    def __init__(
        self,
        *,
        device: Optional[str | torch.device] = None,
        reset: bool = False,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        self.rawdata_path = self.RAWDATA_DIR / self.rawdata_file
        self.device = device

        if not inspect.isabstract(self):
            if reset:
                self.RAWDATA_DIR.rmdir()

            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)

            if initialize:
                self.download()

        self.components = LazyDict(
            {k: self.get_component for k in self.component_files}
        )

    @classmethod
    def from_zipfile(
        cls, zipfile: str | Path, /, *args: Any, **kwargs: Any
    ) -> Any:  # FIXME: Return Self PreTrainedModel:
        r"""Create model from zipfile."""
        path = Path(zipfile)
        if path.suffix != ".zip":
            raise ValueError(f"{path=} is not a zipfile!")

        if not path.is_relative_to(Path.cwd()):
            # print(f"No path provided assuming current working directory.")
            path = Path.cwd() / path

        if not path.exists():
            raise ValueError(f"{path=} does not exist!")

        # FIXME: This is an awful hack!
        obj = cls.__new__(cls)
        obj.RAWDATA_DIR = path.parent
        obj.RAWDATA_HASH = None
        obj.rawdata_file = path.name
        obj.__init__(*args, **kwargs)  # type: ignore[misc]
        return obj

    @classmethod
    def from_url(
        cls, url: str, /, *args: Any, **kwargs: Any
    ) -> Any:  # FIXME: Return Self PreTrainedModel
        r"""Create model from url."""
        # download the file into the model directory
        fname = url.split("/")[-1]
        path = cls.RAWDATA_DIR / fname

        if path.exists():
            warnings.warn(f"{Path=} already exists, skipping download.")
        else:
            download(url, path)

        # FIXME: This is an awful hack!
        obj = cls.__new__(cls)
        obj.RAWDATA_HASH = None
        obj.rawdata_file = path.name
        obj.__init__(*args, **kwargs)  # type: ignore[misc]
        return obj

    @classmethod
    def from_checkpoint(cls, key: str, /) -> Any:  # FIXME: Use typing.Self
        r"""Create model from checkpoint."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, key: str) -> Any:
        return self.components[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __repr__(self) -> str:
        return repr_mapping(self.components, repr_fun=repr_short, wrapped=self)

    @cached_property
    def model(self) -> torch.nn.Module:
        r"""Return the model."""
        return self.components["model"]

    @cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        r"""Return the optimizer."""
        return self.components["optimizer"]

    @cached_property
    def hyperparameters(self) -> dict[str, Any]:
        r"""Return the hyperparameters."""
        return self.components["hyperparameters"]

    @cached_property
    def encoder(self, /) -> BaseEncoder:
        r"""Return the encoder."""
        return self.components["encoder"]

    @cached_property
    def lr_scheduler(self) -> TorchLRScheduler:
        r"""Return the learning rate scheduler."""
        return self.components["lr_scheduler"]

    # @cached_property
    # def components(self) -> dict[str, Any]:
    #     r"""Extract all components."""
    #     return LazyDict({k: self.get_component for k in self.component_files})

    # @property
    # @abstractmethod
    # def component_files(self) -> Mapping[str, str]:
    #     r"""Return filename."""

    # @property
    # @abstractmethod
    # def rawdata_file(self) -> str:
    #     r"""Return filename."""

    def rawdata_file_exist(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.rawdata_path)

    def get_component(self, component: str, /) -> Any:
        r"""Extract a model component."""
        if not self.rawdata_file_exist():
            self.LOGGER.debug("Obtaining rawdata file!")
            self.download(validate=True)

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
            raise FileNotFoundError(f"File {file.name!r} does not exist!")

        filehash = sha256(file.read_bytes()).hexdigest()

        if reference is None:
            warnings.warn(
                f"File {file.name!r} cannot be validated as no hash is stored in {self.__class__}."
                f"The filehash is {filehash!r}."
            )

        elif isinstance(reference, str):
            if filehash != reference:
                warnings.warn(
                    f"File {file.name!r} failed to validate!"
                    f"File hash {filehash!r} does not match reference {reference!r}."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            self.LOGGER.info(
                f"File {file.name!r} validated successfully {filehash=!r}."
            )

        elif isinstance(reference, Mapping):
            if not (file.name in reference) ^ (file.stem in reference):
                warnings.warn(
                    f"File {file.name!r} cannot be validated as it is not contained in {reference}."
                    f"The filehash is {filehash!r}."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            elif file.name in reference and filehash != reference[file.name]:
                warnings.warn(
                    f"File {file.name!r} failed to validate!"
                    f"File hash {filehash!r} does not match reference {reference[file.name]!r}."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            elif file.stem in reference and filehash != reference[file.stem]:
                warnings.warn(
                    f"File {file.name!r} failed to validate!"
                    f"File hash {filehash!r} does not match reference {reference[file.stem]!r}."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            else:
                self.LOGGER.info(
                    f"File {file.name!r} validated successfully {filehash=!r}."
                )
        else:
            raise TypeError(f"Unsupported type for {reference=}.")

        self.LOGGER.debug("Finished validating file.")
