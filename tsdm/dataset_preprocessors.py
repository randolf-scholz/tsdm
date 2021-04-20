import logging
import zipfile

import numpy as np
import pandas

from .config import RAWDATADIR, DATASETDIR

logger = logging.getLogger(__name__)


def preprocess_electricity():
    logger.info("Preprocessing Electricity Dataset")
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
    logger.info("Finished Preprocessing Electricity Dataset")


def preprocess_traffic():
    raise NotImplementedError


def preprocess_human_activity():
    raise NotImplementedError


def preprocess_air_quality_multisite():
    raise NotImplementedError


def preprocess_air_quality():
    raise NotImplementedError


def preprocess_household_consumptions():
    raise NotImplementedError


def preprocess_character_trajectories():
    raise NotImplementedError


def preprocess_mujoco():
    raise NotImplementedError


def preprocess_m3():
    raise NotImplementedError


def preprocess_uwave():
    raise NotImplementedError


def preprocess_physionet2012():
    raise NotImplementedError


def preprocess_physionet2019():
    raise NotImplementedError


def preprocess_ushcn():
    raise NotImplementedError


def preprocess_m4():
    raise NotImplementedError


def preprocess_m5():
    raise NotImplementedError


def preprocess_tourism1():
    raise NotImplementedError


def preprocess_tourism2():
    raise NotImplementedError


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
    """
    Pre-Processes Dataset according to built-in Routine

    Parameters
    ----------
    dataset: str

    Returns
    -------

    """
    return dataset_preprocessors[dataset]()
