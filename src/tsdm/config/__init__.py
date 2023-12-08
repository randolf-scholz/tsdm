r"""Configuration Options."""

__all__ = [
    # CONSTANTS
    "CONFIG",
    "PROJECT",
    # Classes
    "Config",
    "Project",
    # Functions
    "generate_folders",
    "get_package_structure",
]

from tsdm.config._config import (
    CONFIG,
    PROJECT,
    Config,
    Project,
    generate_folders,
    get_package_structure,
)
