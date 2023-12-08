r"""Time Series Datasets and Models (TSDM).

TODO: rewrite introduction
"""

__all__ = [
    # Constants
    "__version__",
    # Functions
    "info",
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


def info(obj: object | None = None) -> None:
    """Open the help page for the given object in a browser."""
    import inspect  # pylint: disable=import-outside-toplevel
    import webbrowser  # pylint: disable=import-outside-toplevel

    url = config.PROJECT.DOC_URL
    if obj is None:
        webbrowser.open(url)
        return

    pkg = inspect.getmodule(obj)
    if pkg is None:
        pkg = inspect.getmodule(type(obj))
    if pkg is None:
        raise ValueError("Could not determine package of object!")
    if not pkg.__name__.startswith(__package__):
        raise ValueError(f"Object does not belong to {__package__!r}!")
    webbrowser.open(f"{url}/apidoc/{pkg.__name__}.html")
