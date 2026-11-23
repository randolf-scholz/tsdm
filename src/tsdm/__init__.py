r"""Time Series Datasets and Models (TSDM)."""

__all__ = [
    # Constants
    "__version__",
    # Sub-Modules
    "backend",
    "config",
    "datasets",
    "encoders",
    "linalg",
    "logutils",
    "losses",
    "plotting",
    "pretrained",
    "random",
    "stats",
    "tasks",
    "timeseries",
    "types",
    "utils",
]


from .constants import __version__
lazy from . import (
    backend,
    config,
    datasets,
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
