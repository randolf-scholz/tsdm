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
    "DATASETS",
    "MODELS",
    "HASHES",
    "HOMEDIR",
    "BASEDIR",
    "LOGDIR",
    "MODELDIR",
    "DATASETDIR",
    "RAWDATADIR",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "conf",
    # Classes
    "Config",
]

import logging

from tsdm.config._config import CONFIG  # CONSTANTS; Classes
from tsdm.config._config import (
    BASEDIR,
    DATASETDIR,
    DATASETS,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    HASHES,
    HOMEDIR,
    LOGDIR,
    MODELDIR,
    MODELS,
    RAWDATADIR,
    Config,
    conf,
)

__logger__ = logging.getLogger(__name__)
