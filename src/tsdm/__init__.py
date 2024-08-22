r"""Time Series Datasets and Models (TSDM)."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Modules
    "backend",
    "config",
    "data",
    "datasets",
    "encoders",
    "linalg",
    "logutils",
    "metrics",
    "models",
    "optimizers",
    "random",
    "tasks",
    "timeseries",
    "types",
    "utils",
    "viz",
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

from tsdm import (
    backend,
    config,
    data,
    datasets,
    encoders,
    linalg,
    logutils,
    metrics,
    models,
    optimizers,
    random,
    tasks,
    timeseries,
    types,
    utils,
    viz,
)
