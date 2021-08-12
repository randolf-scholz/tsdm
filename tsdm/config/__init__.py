r"""Configuration Options.

Content:
  - config.yaml
  - datasets.yaml
  - models.yaml
  - hashes.yaml
"""

from importlib import resources
import logging
from pathlib import Path
from typing import Final

import yaml

from tsdm.config import config_files

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = [
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
]

with resources.path(config_files, "config.yaml") as file:
    with open(file, "r") as fname:
        CONFIG = yaml.safe_load(fname)
        r"""Dictionary containing basic configuration of TSDM."""

with resources.path(config_files, "models.yaml") as file:
    with open(file, "r") as fname:
        MODELS = yaml.safe_load(fname)
        r"""Dictionary containing sources of the available models."""

with resources.path(config_files, "datasets.yaml") as file:
    with open(file, "r") as fname:
        DATASETS = yaml.safe_load(fname)
        r"""Dictionary containing sources of the available datasets."""

with resources.path(config_files, "hashes.yaml") as file:
    with open(file, "r") as fname:
        HASHES = yaml.safe_load(fname)
        r"""Dictionary containing hash values for both models and datasets."""

HOMEDIR = Path.home()
r"""The users home directory."""

BASEDIR = HOMEDIR.joinpath(CONFIG["basedir"])
r"""Root directory for tsdm storage."""

LOGDIR = BASEDIR.joinpath(CONFIG["logdir"])
r"""Path where logfiles are stored."""

MODELDIR = BASEDIR.joinpath(CONFIG["modeldir"])
r"""Path where imported models are stored."""

DATASETDIR = BASEDIR.joinpath(CONFIG["datasetdir"])
r"""Path where preprocessed datasets are stored."""

RAWDATADIR = BASEDIR.joinpath(CONFIG["rawdatadir"])
r"""Path where raw imported datasets are stored."""

LOGDIR.mkdir(parents=True, exist_ok=True)
# logging.basicConfig(
#     filename=str(LOGDIR.joinpath("example.log")),
#     filemode="w",
#     format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s, (%(filename)s:%(lineno)s)",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.INFO)

logger.info("Available Models: %s", set(MODELS))
logger.info("Available Datasets: %s", set(DATASETS))


def generate_folders(d: dict, current_path: Path):
    """Create nested folder structure based on nested dictionary keys.

    source: `StackOverflow <https://stackoverflow.com/a/22058144/9318372>`_

    Parameters
    ----------
    current_path: Path
    d: dict

    Returns
    -------
    None
    """
    for directory in d:
        path = current_path.joinpath(directory)
        if d[directory] is None:
            logger.info("creating folder %s", path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            generate_folders(d[directory], path)


# logger.info(F"Found config files: {set(resources.contents('config_files'))}")
logger.info("Initializing Folder Structure")
generate_folders(CONFIG["folders"], BASEDIR)
