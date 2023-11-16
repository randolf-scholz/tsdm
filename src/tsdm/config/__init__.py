r"""Configuration Options.

Content:

- config.yaml
- dataset.yaml
- models.yaml
- hashes.yaml
"""

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

from typing_extensions import Final

from tsdm.config._config import Config, Project, generate_folders, get_package_structure

PROJECT: Final[Project] = Project()
"""Project configuration."""

CONFIG: Final[Config] = Config()
"""Configuration Class."""

del Final
