r"""Time Series Datasets and Models (TSDM).

TODO: rewrite introduction
"""

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
    "types",
    "utils",
    "viz",
]

from importlib import metadata

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
    types,
    utils,
    viz,
)

# single-source version
try:
    __version__ = metadata.version(__package__ or __name__)
    r"""The version number of the `tsdm` package."""
except metadata.PackageNotFoundError:
    __version__ = "unknown"
    r"""The version number of the `tsdm` package."""
finally:
    del metadata
