import yaml
import logging
from pathlib import Path
from importlib import resources
from . import config_files

with resources.path(config_files, "config.yaml") as file:
    with open(file, "r") as fname:
        CONFIG = yaml.safe_load(fname)


with resources.path(config_files, "models.yaml") as file:
    with open(file, "r") as fname:
        MODELS = yaml.safe_load(fname)

with resources.path(config_files, "datasets.yaml") as file:
    with open(file, "r") as fname:
        DATASETS = yaml.safe_load(fname)

with resources.path(config_files, "hashes.yaml") as file:
    with open(file, "r") as fname:
        HASHES = yaml.safe_load(fname)


HOMEDIR    = Path.home()
BASEDIR    = HOMEDIR.joinpath(CONFIG['basedir'])
LOGDIR     = BASEDIR.joinpath(CONFIG['logdir'])
MODELDIR   = BASEDIR.joinpath(CONFIG['modeldir'])
DATASETDIR = BASEDIR.joinpath(CONFIG['datasetdir'])
RAWDATADIR = BASEDIR.joinpath(CONFIG['rawdatadir'])
LOGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOGDIR.joinpath("example.log")),
    filemode="w",
    format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s",  # (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)

logger = logging.getLogger(__name__)


def generate_folders(d: dict or str, current_path: Path) -> None:
    """
    Creates nested folder structure based on nested dictionary keys
    source: `StackOverflow <https://stackoverflow.com/a/22058144/9318372>`_

    Parameters
    ----------
    current_path: Path
    d: dict

    Returns
    -------
    None
    """
    for direc in d:
        path = current_path.joinpath(direc)
        if d[direc] is None:
            logging.info(F"creating folder {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            generate_folders(d[direc], path)
    return


logger.info(F"Found config files: {set(resources.contents('config_files'))}")
logger.info("Initializing Folder Structure")
generate_folders(CONFIG['folders'], BASEDIR)

AVAILABLE_MODELS = set().union(*[set(MODELS[source]) for source in MODELS])
logger.info(F"{AVAILABLE_MODELS=}")

AVAILABLE_DATASETS = set().union(*[set(DATASETS[source]) for source in DATASETS])
logger.info(F"{AVAILABLE_DATASETS=}")
