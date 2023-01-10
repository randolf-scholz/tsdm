r"""TSDM Configuration.

# TODO: There must be a better way to handle global config
"""

from __future__ import annotations

__all__ = [
    # CONSTANTS
    "CONFIG",
    "PROJECT",
    # Classes
    "Project",
    "Config",
    # Functions
    "get_package_structure",
]

import logging
import os
from importlib import import_module, resources
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import torch
import yaml

from tsdm.config import config_files

__logger__ = logging.getLogger(__name__)


def get_package_structure(root_module: ModuleType, /) -> dict[str, Any]:
    r"""Create nested dictionary of the package structure."""
    d = {}
    for name in dir(root_module):
        attr = getattr(root_module, name)
        if isinstance(attr, ModuleType):
            # check if it is a subpackage
            if (
                attr.__name__.startswith(root_module.__name__)
                and attr.__package__ != root_module.__package__
                and attr.__package__ is not None
            ):
                d[attr.__package__] = get_package_structure(attr)
    return d


def generate_folders(d: dict, current_path: Path) -> None:
    r"""Create nested folder structure based on nested dictionary index.

    References
    ----------
    `StackOverflow <https://stackoverflow.com/a/22058144/9318372>`_
    """
    for directory in d:
        path = current_path.joinpath(directory)
        if d[directory] is None:
            __logger__.debug("creating folder %s", path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            generate_folders(d[directory], path)


class Config:
    r"""Configuration Interface."""

    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r"""The default `torch` device to use."""
    DEFAULT_DTYPE = torch.float32
    r"""The default `torch` datatype to use."""
    HOMEDIR: Path
    r"""The users home directory."""
    BASEDIR: Path
    r"""Root directory for tsdm storage."""
    LOGDIR: Path
    r"""Path where logfiles are stored."""
    MODELDIR: Path
    r"""Path where imported models are stored."""
    DATASETDIR: Path
    r"""Path where preprocessed dataset are stored."""
    RAWDATADIR: Path
    r"""Path where raw imported dataset are stored."""

    _autojit: bool = True

    def __init__(self):
        r"""Initialize the configuration."""
        # TODO: Should be initialized by an init/toml file.
        os.environ["TSDM_AUTOJIT"] = "True"
        self._autojit: bool = True
        self.HOMEDIR = Path.home()
        self.BASEDIR = self.HOMEDIR.joinpath(self.CONFIG_FILE["basedir"])
        self.LOGDIR = self.BASEDIR.joinpath(self.CONFIG_FILE["logdir"])
        self.MODELDIR = self.BASEDIR.joinpath(self.CONFIG_FILE["modeldir"])
        self.DATASETDIR = self.BASEDIR.joinpath(self.CONFIG_FILE["datasetdir"])
        self.RAWDATADIR = self.BASEDIR.joinpath(self.CONFIG_FILE["rawdatadir"])

        # further initialization
        self.LOGDIR.mkdir(parents=True, exist_ok=True)
        __logger__.info("Available Models: %s", set(self.MODELS))
        __logger__.info("Available Datasets: %s", set(self.DATASETS))
        __logger__.debug("Initializing folder structure")
        generate_folders(self.CONFIG_FILE["folders"], self.BASEDIR)
        __logger__.debug("Created folder structure in %s", self.BASEDIR)

    @property
    def autojit(self) -> bool:
        r"""Whether to automatically jit-compile the models."""
        return self._autojit

    @autojit.setter
    def autojit(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._autojit = bool(value)
        os.environ["TSDM_AUTOJIT"] = str(value)

    @property
    def CONFIG_FILE(self) -> dict:
        r"""Return dictionary containing basic configuration of TSDM."""
        with resources.open_text(config_files, "config.yaml") as file:
            # with open(file, "r", encoding="utf8") as fname:
            return yaml.safe_load(file)

    @property
    def MODELS(self) -> dict:
        r"""Dictionary containing sources of the available models."""
        with resources.open_text(config_files, "models.yaml") as file:
            return yaml.safe_load(file)

    @property
    def DATASETS(self) -> dict:
        r"""Dictionary containing sources of the available datasets."""
        with resources.open_text(config_files, "datasets.yaml") as file:
            return yaml.safe_load(file)

    @property
    def HASHES(self) -> dict:
        r"""Dictionary containing hashes of the available datasets."""
        with resources.open_text(config_files, "hashes.yaml") as file:
            return yaml.safe_load(file)


class Project:
    """Holds Project related data."""

    @property
    def NAME(self) -> str:
        r"""Get project name."""
        return self.ROOT_PACKAGE.__name__

    @property
    def ROOT_PACKAGE(self) -> ModuleType:
        r"""Get project root package."""
        hierarchy = __package__.split(".")
        return import_module(hierarchy[0])

    @property
    def ROOT_PATH(self) -> Path:
        r"""Return the root directory."""
        assert len(self.ROOT_PACKAGE.__path__) == 1
        return Path(self.ROOT_PACKAGE.__path__[0])

    @property
    def TESTS_PATH(self) -> Path:
        r"""Return the test directory."""
        tests_path = self.ROOT_PATH.parent.parent / "tests"

        if self.ROOT_PATH.parent.stem != "src":
            raise ValueError(
                f"This seems to be an installed version of {self.NAME},"
                f" as {self.ROOT_PATH} is not in src/*"
            )
        if not tests_path.exists():
            raise ValueError(f"Tests directory {tests_path} does not exist!")
        return tests_path

    @property
    def SOURCE_PATH(self) -> Path:
        r"""Return the source directory."""
        source_path = self.ROOT_PATH.parent.parent / "src"

        if self.ROOT_PATH.parent.stem != "src":
            raise ValueError(
                f"This seems to be an installed version of {self.NAME},"
                f" as {self.ROOT_PATH} is not in src/*"
            )
        if not source_path.exists():
            raise ValueError(f"Source directory {source_path} does not exist!")
        return source_path

    def make_test_folders(self, dry_run: bool = True) -> None:
        r"""Make the tests folder if it does not exist."""
        package_structure = get_package_structure(self.ROOT_PACKAGE)

        def flattened(d: dict[str, Any], /) -> list[str]:
            r"""Flatten nested dictionary."""
            return list(d) + list(chain.from_iterable(map(flattened, d.values())))

        for package in flattened(package_structure):
            test_package_path = PROJECT.TESTS_PATH / package.replace(".", "/")
            test_package_init = test_package_path / "__init__.py"

            if not test_package_path.exists():
                if dry_run:
                    print(f"Dry-Run: Creating {test_package_path}")
                else:
                    print("Creating {test_package_path}")
                    test_package_path.mkdir(parents=True, exist_ok=True)
            if not test_package_path.exists():
                if dry_run:
                    print(f"Dry-Run: Creating {test_package_init}")
                else:
                    raise RuntimeError(f"Creation of {test_package_path} failed!")
            elif not test_package_init.exists():
                if dry_run:
                    print(f"Dry-Run: Creating {test_package_init}")
                else:
                    print(f"Creating {test_package_init}")
                    with open(test_package_init, "w", encoding="utf8") as file:
                        file.write(f'"""Tests for {package}."""\n')

        if dry_run:
            print("Pass option `dry_run=False` to actually create the folders.")


CONFIG: Final[Config] = Config()
r"""The unique `~tsdm.config.Config` instance used to configure `tsdm`."""

PROJECT: Final[Project] = Project()
"""Project singleton."""

# logging.basicConfig(
#     filename=str(LOGDIR.joinpath("example.log")),
#     filemode="w",
#     format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s, (%(filename)s:%(lineno)s)",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.INFO)
