import logging
import zipfile

import numpy as np
import pandas

from .config import DATASETDIR, RAWDATADIR
from .dataset_io import dataset_available, download_dataset

logger = logging.getLogger(__name__)


def clean_electricity():
    logger.info("Cleaning Electricity Dataset")
    rawdata_path = RAWDATADIR.joinpath('electricity')
    dataset_path = DATASETDIR.joinpath('electricity')
    fname = "LD2011_2014.txt"
    files = zipfile.ZipFile(rawdata_path.joinpath(fname + ".zip"))
    files.extract(fname, path=dataset_path)
    logger.info("Finished Extracting Electricity Dataset")
    df = pandas.read_csv(dataset_path.joinpath(fname),
                         sep=";", decimal=",", parse_dates=[0], index_col=0, dtype=np.float64)
    df = df.rename_axis(index="time", columns="client")
    df.name = "electricity"
    df.to_hdf(dataset_path.joinpath("electricity.h5"), key="electricity")
    df.to_csv(dataset_path.joinpath("electricity.csv"))
    dataset_path.joinpath(fname).unlink()
    logger.info("Finished Cleaning Electricity Dataset")


def clean_traffic():
    raise NotImplementedError


def clean_human_activity():
    raise NotImplementedError


def clean_air_quality_multisite():
    raise NotImplementedError


def clean_air_quality():
    raise NotImplementedError


def clean_household_consumptions():
    raise NotImplementedError


def clean_character_trajectories():
    raise NotImplementedError


def clean_mujoco():
    raise NotImplementedError


def clean_m3():
    raise NotImplementedError


def clean_uwave():
    raise NotImplementedError


def clean_physionet2012():
    raise NotImplementedError


def clean_physionet2019():
    raise NotImplementedError


def clean_ushcn():
    raise NotImplementedError


def clean_m4():
    raise NotImplementedError


def clean_m5():
    raise NotImplementedError


def clean_tourism1():
    raise NotImplementedError


def clean_tourism2():
    raise NotImplementedError


dataset_cleaners = {
    'electricity'            : clean_electricity,
    'traffic'                : clean_traffic,
    'human activity'         : clean_human_activity,
    'air quality multi-site' : clean_air_quality_multisite,
    'air quality'            : clean_air_quality,
    'household consumption'  : clean_household_consumptions,
    'character trajectories' : clean_character_trajectories,
    'MuJoCo'                 : clean_mujoco,
    'M3'                     : clean_m3,
    'UWAVE'                  : clean_uwave,
    'Physionet 2012'         : clean_physionet2012,
    'Physionet 2019'         : clean_physionet2019,
    'USHCN'                  : clean_ushcn,
    'M4'                     : clean_m4,
    'M5'                     : clean_m5,
    'tourism1'               : clean_tourism1,
    'tourism2'               : clean_tourism2,
}


def clean_dataset(dataset: str):
    """
    Pre-Processes Dataset according to built-in Routine

    Parameters
    ----------
    dataset: str

    Returns
    -------

    """

    assert dataset_available(dataset)

    if not DATASETDIR.joinpath(dataset).exists():
        download_dataset(dataset)

    return dataset_cleaners[dataset]()
