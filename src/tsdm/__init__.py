r"""Time Series Datasets and Models (TSDM)."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Modules
    "backend",
    "config",
    "datatools",
    "datasets",
    "encoders",
    "linalg",
    "logutils",
    "losses",
    "pretrained",
    "random",
    "stats",
    "tasks",
    "timeseries",
    "types",
    "utils",
    "plotting",
]

from importlib import metadata

try:  # single-source version
    __version__ = metadata.version(__package__ or __name__)
    r"""The version number of the `tsdm` package."""
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    r"""The version number of the `tsdm` package."""
finally:
    del metadata

from . import config

config.initialize_directories()

from . import (  # ruff: ignore[E402]  # Initialize storage before importing dependent modules.
    backend,
    datasets,
    datatools,
    encoders,
    linalg,
    logutils,
    losses,
    plotting,
    pretrained,
    random,
    stats,
    tasks,
    timeseries,
    types,
    utils,
)
