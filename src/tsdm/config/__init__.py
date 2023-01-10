r"""Configuration Options.

Content:

- config.yaml
- dataset.yaml
- models.yaml
- hashes.yaml
"""

__all__ = [
    # CONSTANTS
    "PROJECT",
    "CONFIG",
    # Classes
    "Project",
    "Config",
    # Functions
    "get_package_structure",
    "generate_folders",
]

from typing import Final

from tsdm.config._config import Config, Project, generate_folders, get_package_structure

PROJECT: Final[Project] = Project()
"""Project configuration."""

CONFIG: Final[Config] = Config()
"""Configuration Class."""

del Final
