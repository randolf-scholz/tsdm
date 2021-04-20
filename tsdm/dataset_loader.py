import yaml
import logging
from pathlib import Path
import subprocess
import pandas
import zipfile
import xarray
import numpy as np

with open("config/config.yaml") as fname:
    CONFIG = yaml.safe_load(fname)

with open("config/models.yaml") as fname:
    MODELS = yaml.safe_load(fname)

with open("config/datasets.yaml") as fname:
    DATASETS = yaml.safe_load(fname)

with open("config/hashes.yaml") as fname:
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


def load_electricity():
    dataset = DATASETDIR.joinpath("electricity/electricity.h5")
    if not dataset.exists():
        download_dataset('electricity')
        preprocess_electricity()

    return pandas.read_hdf(dataset, key="electricity")


def load_traffic():
    pass


def load_human_activity():
    pass


def load_air_quality_multisite():
    pass


def load_air_quality():
    pass


def load_household_consumptions():
    pass


def load_character_trajectories():
    pass


def load_mujoco():
    pass


def load_m3():
    pass


def load_uwave():
    pass


def load_physionet2012():
    pass


def load_physionet2019():
    pass


def load_ushcn():
    pass


def load_m4():
    pass


def load_m5():
    pass


def load_tourism1():
    pass


def load_tourism2():
    pass


dataset_loaders = {
    'electricity'            : load_electricity,
    'traffic'                : load_traffic,
    'human activity'         : load_human_activity,
    'air quality multi-site' : load_air_quality_multisite,
    'air quality'            : load_air_quality,
    'household consumption'  : load_household_consumptions,
    'character trajectories' : load_character_trajectories,
    'MuJoCo'                 : load_mujoco,
    'M3'                     : load_m3,
    'UWAVE'                  : load_uwave,
    'Physionet 2012'         : load_physionet2012,
    'Physionet 2019'         : load_physionet2019,
    'USHCN'                  : load_ushcn,
    'M4'                     : load_m4,
    'M5'                     : load_m5,
    'tourism1'               : load_tourism1,
    'tourism2'               : load_tourism2,
}


def load_dataset(dataset: str):
    return dataset_loaders[dataset]()
