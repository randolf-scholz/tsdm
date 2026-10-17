r"""TSDM Configuration."""
# ruff: file-ignore[N802]

__all__ = [
    # Classes
    "Config",
    "Project",
    # Constants
    "CONFIG",
    "PROJECT",
]

import logging
import os
from enum import StrEnum
from functools import cached_property
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Final, final


@final
class Config:
    r"""Configuration Interface."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the class."""
    GENERATING_DOCS: ClassVar[bool] = bool(os.environ.get("TSDM_DOCS"))
    r"""Whether the documentation is being generated."""

    ROOT_DIR: Path
    r"""Root directory for tsdm storage."""
    LOG_DIR: Path
    r"""Path where logfiles are stored."""
    MODEL_DIR: Path
    r"""Path where imported models are stored."""
    DATASET_DIR: Path
    r"""Path where imported dataset are stored."""
    DEFAULT_HASH_METHOD = "sha256"
    r"""Default hash method for file integrity checks."""
    DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024
    r"""Default chunk size for file integrity checks."""

    class DATASET_PATHS(StrEnum):
        r"""Keys for dataset schema."""

        ROOT = "root"
        r"""Root data directory."""
        RAWDATA = "raw"
        r"""Raw data directory."""
        PROCESSED = "processed"
        r"""Processed data directory."""
        METADATA = "metadata"
        r"""Metadata directory."""

    @cached_property
    def DEFAULT_CONFIG(self) -> dict[str, Any]:
        r"""Dictionary containing basic configuration of TSDM."""
        # dataset schema:
        # ~/.tsdm/datasets/<dataset>/raw/<version>/... = immutable downloads (as received)
        # ~/.tsdm/datasets/<dataset>/processed/<version>/... = reproducible cleaned artifacts (e.g. parquet)
        # ~/.tsdm/datasets/<dataset>/meta/<version>/... = hashes, manifests, schemas, logs (optional but useful)
        # ~/.tsdm/datasets/<dataset>/tmp/ = scratch space (safe to delete)
        return {
            "root_dir"    : "~/.tsdm",
            "log_dir"     : "logs",
            "model_dir"   : "models",
            "dataset_dir" : "datasets",
            "dataset_schema": {
                self.DATASET_PATHS.ROOT    : "root",
                self.DATASET_PATHS.RAWDATA : "raw",
                self.DATASET_PATHS.PROCESSED : "processed",
                self.DATASET_PATHS.METADATA    : "meta",
            },
            "folders"    : ["datasets", "models", "logs"],
        }  # fmt: skip

    def __init__(self) -> None:
        r"""Initialize the configuration."""
        # TODO: Should be initialized by an init/toml file.
        self.ROOT_DIR = Path(self.DEFAULT_CONFIG["root_dir"])
        if not self.GENERATING_DOCS:
            self.ROOT_DIR = self.ROOT_DIR.expanduser().absolute()

        self.LOG_DIR = self.ROOT_DIR / self.DEFAULT_CONFIG["log_dir"]
        self.MODEL_DIR = self.ROOT_DIR / self.DEFAULT_CONFIG["model_dir"]
        self.DATASET_DIR = self.ROOT_DIR / self.DEFAULT_CONFIG["dataset_dir"]

        # further initialization
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGGER.debug("Initializing folder structure")
        for folder in self.DEFAULT_CONFIG["folders"]:
            (self.ROOT_DIR / folder).mkdir(parents=True, exist_ok=True)
        self.LOGGER.debug("Created folder structure in %s", self.ROOT_DIR)


@final
class Project:
    r"""Holds Project related data."""

    @cached_property
    def NAME(self) -> str:
        r"""Project name."""
        return self.ROOT_PACKAGE.__name__

    @cached_property
    def ROOT_PACKAGE(self) -> ModuleType:
        r"""Project root package."""
        if __package__ is None:
            raise ValueError(f"Unexpected package: {__package__=}")
        hierarchy = __package__.split(".")
        return import_module(hierarchy[0])

    @cached_property
    def ROOT_PATH(self) -> Path:
        r"""Root directory."""
        if len(self.ROOT_PACKAGE.__path__) != 1:
            raise ValueError(f"Unexpected path: {self.ROOT_PACKAGE.__path__=}")

        path = Path(self.ROOT_PACKAGE.__path__[0])

        if path.parent.stem != "src":
            raise ValueError(
                f"This seems to be an installed version of {self.NAME},"
                f" as {path} is not in src/*"
            )
        return path.parent.parent

    @cached_property
    def DOCS_PATH(self) -> Path:
        r"""`docs` directory."""
        docs_path = self.ROOT_PATH / "docs"
        if not docs_path.exists():
            raise ValueError(f"Docs directory {docs_path} does not exist!")
        return docs_path

    @cached_property
    def SOURCE_PATH(self) -> Path:
        r"""Source directory."""
        source_path = self.ROOT_PATH / "src"
        if not source_path.exists():
            raise ValueError(f"Source directory {source_path} does not exist!")
        return source_path

    @cached_property
    def TESTS_PATH(self) -> Path:
        r"""Test directory."""
        tests_path = self.ROOT_PATH / "tests"
        if not tests_path.exists():
            raise ValueError(f"Tests directory {tests_path} does not exist!")
        return tests_path

    @cached_property
    def TEST_RESULTS_PATH(self) -> Path:
        r"""Test `results` directory."""
        return self.TESTS_PATH / ".results"

    @cached_property
    def RESULTS_DIR(self) -> dict[str | Path, Path]:
        r"""`results` directory."""

        class ResultsDir(dict):
            r"""Results directory."""

            TEST_RESULTS_PATH = self.TEST_RESULTS_PATH

            def __setitem__(self, key: str | Path, value: Path, /) -> None:
                raise RuntimeError("ResultsDir is read-only!")

            def __getitem__(self, key: str | Path, /) -> None:
                if key not in self:
                    path = self.TEST_RESULTS_PATH / Path(key).stem
                    path.mkdir(parents=True, exist_ok=True)
                    super().__setitem__(key, path)
                return super().__getitem__(key)

        return ResultsDir()


# region CONSTANTS ---------------------------------------------------------------------
CONFIG: Final[Config] = Config()
r"""Configuration Class."""

PROJECT: Final[Project] = Project()
r"""Project configuration."""
# endregion CONSTANTS ------------------------------------------------------------------
