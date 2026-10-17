r"""TSDM configuration, read once at import time.

Set ``TSDM_ROOT_DIR`` before importing tsdm to override the storage root.
"""

__all__ = [
    # Classes
    "DATASET_PATHS",
    # Constants
    "GENERATING_DOCS",
    "ROOT_DIR",
    "LOG_DIR",
    "MODEL_DIR",
    "DATASET_DIR",
    "DEFAULT_HASH_METHOD",
    "DEFAULT_CHUNK_SIZE",
    # Functions
    "initialize_directories",
]

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Final

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

GENERATING_DOCS: Final[bool] = bool(os.environ.get("TSDM_DOCS"))
r"""Whether the documentation is being generated."""

_root_dir = Path(os.environ.get("TSDM_ROOT_DIR", "~/.tsdm"))
ROOT_DIR: Final[Path] = (
    _root_dir if GENERATING_DOCS else _root_dir.expanduser().absolute()
)
r"""Root directory for tsdm storage."""
del _root_dir

LOG_DIR: Final[Path] = ROOT_DIR / "logs"
r"""Path where logfiles are stored."""
MODEL_DIR: Final[Path] = ROOT_DIR / "models"
r"""Path where imported models are stored."""
DATASET_DIR: Final[Path] = ROOT_DIR / "datasets"
r"""Path where imported datasets are stored."""
DEFAULT_HASH_METHOD: Final[str] = "sha256"
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


def initialize_directories() -> None:
    r"""Create the configured storage directories, allowing existing directories."""
    _LOGGER.debug("Initializing folder structure")
    for path in (LOG_DIR, MODEL_DIR, DATASET_DIR):
        path.mkdir(parents=True, exist_ok=True)
    _LOGGER.debug("Created folder structure in %s", ROOT_DIR)
