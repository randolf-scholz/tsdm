r"""TSDM configuration, read once at import time.

The first import of `tsdm` creates the storage directories. Set `TSDM_ROOT_DIR`
before importing `tsdm` to override the storage root.
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
]

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Final

GENERATING_DOCS: Final[bool] = bool(os.environ.get("TSDM_DOCS"))
r"""Whether the documentation is being generated."""

ROOT_DIR: Final[Path] = (
    Path(os.environ.get("TSDM_ROOT_DIR", "~/.tsdm"))
    if GENERATING_DOCS
    else Path(os.environ.get("TSDM_ROOT_DIR", "~/.tsdm")).expanduser().absolute()
)
r"""Root directory for tsdm storage."""
LOG_DIR: Final[Path] = ROOT_DIR / "logs"
r"""Path where logfiles are stored."""
MODEL_DIR: Final[Path] = ROOT_DIR / "models"
r"""Path where imported models are stored."""
DATASET_DIR: Final[Path] = ROOT_DIR / "datasets"
r"""Path where imported datasets are stored."""


class DATASET_PATHS(StrEnum):
    r"""Names of the storage directories associated with a dataset."""

    ROOT = "root"
    r"""Root data directory."""
    RAWDATA = "raw"
    r"""Raw data directory."""
    PROCESSED = "processed"
    r"""Processed data directory."""
    METADATA = "metadata"
    r"""Metadata directory."""


DEFAULT_HASH_METHOD: Final[str] = "sha256"
r"""Default hash method for file integrity checks."""
DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024
r"""Default chunk size in bytes for file integrity checks."""


_LOGGER = logging.getLogger(__name__)
_LOGGER.debug("Initializing folder structure")
for _directory in (LOG_DIR, MODEL_DIR, DATASET_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
_LOGGER.debug("Created folder structure in %s", ROOT_DIR)
del _LOGGER, _directory
