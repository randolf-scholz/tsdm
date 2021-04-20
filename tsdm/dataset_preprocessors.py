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


def preprocess_electricity():
    logging.info("Preprocessing Electricity Dataset")
    rawdata_path = RAWDATADIR.joinpath('electricity')
    dataset_path = DATASETDIR.joinpath('electricity')
    fname = "LD2011_2014.txt"
    files = zipfile.ZipFile(rawdata_path.joinpath(fname + ".zip"))
    files.extract(fname, path=dataset_path)
    logging.info("Finished Extracting Electricity Dataset")
    df = pandas.read_csv(dataset_path.joinpath(fname),
                         sep=";", decimal=",", parse_dates=[0], index_col=0, dtype=np.float64)
    df = df.rename_axis(index="time", columns="client")
    df.name = "electricity"
    df.to_hdf(dataset_path.joinpath("electricity.h5"), key="electricity")
    df.to_csv(dataset_path.joinpath("electricity.csv"))
    dataset_path.joinpath(fname).unlink()
    logging.info("Finished Preprocessing Electricity Dataset")

def preprocess_traffic():
    pass


def preprocess_human_activity():
    pass


def preprocess_air_quality_multisite():
    pass


def preprocess_air_quality():
    pass


def preprocess_household_consumptions():
    pass


def preprocess_character_trajectories():
    pass


def preprocess_mujoco():
    pass


def preprocess_m3():
    pass


def preprocess_uwave():
    pass


def preprocess_physionet2012():
    pass


def preprocess_physionet2019():
    pass


def preprocess_ushcn():
    pass


def preprocess_m4():
    pass


def preprocess_m5():
    pass


def preprocess_tourism1():
    pass


def preprocess_tourism2():
    pass


dataset_preprocessors = {
    'electricity'            : preprocess_electricity,
    'traffic'                : preprocess_traffic,
    'human activity'         : preprocess_human_activity,
    'air quality multi-site' : preprocess_air_quality_multisite,
    'air quality'            : preprocess_air_quality,
    'household consumption'  : preprocess_household_consumptions,
    'character trajectories' : preprocess_character_trajectories,
    'MuJoCo'                 : preprocess_mujoco,
    'M3'                     : preprocess_m3,
    'UWAVE'                  : preprocess_uwave,
    'Physionet 2012'         : preprocess_physionet2012,
    'Physionet 2019'         : preprocess_physionet2019,
    'USHCN'                  : preprocess_ushcn,
    'M4'                     : preprocess_m4,
    'M5'                     : preprocess_m5,
    'tourism1'               : preprocess_tourism1,
    'tourism2'               : preprocess_tourism2,
}


def preprocess_dataset(dataset: str):
    return dataset_preprocessors[dataset]()


preprocess_electricity()
