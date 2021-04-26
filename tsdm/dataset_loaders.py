import logging
import pandas

from .config import DATASETDIR
from .dataset_io import dataset_available, download_dataset
from .dataset_cleaners import clean_dataset

logger = logging.getLogger(__name__)


def load_electricity():
    dataset = DATASETDIR.joinpath("electricity/electricity.h5")
    if not dataset.exists():
        download_dataset('electricity')
        clean_dataset('electricity')
    df = pandas.read_hdf(dataset, key="electricity")
    return df


def load_traffic():
    raise NotImplementedError


def load_human_activity():
    raise NotImplementedError


def load_air_quality_multisite():
    raise NotImplementedError


def load_air_quality():
    raise NotImplementedError


def load_household_consumptions():
    raise NotImplementedError


def load_character_trajectories():
    raise NotImplementedError


def load_mujoco():
    raise NotImplementedError


def load_m3():
    raise NotImplementedError


def load_uwave():
    raise NotImplementedError


def load_physionet2012():
    raise NotImplementedError


def load_physionet2019():
    raise NotImplementedError


def load_ushcn():
    raise NotImplementedError


def load_m4():
    raise NotImplementedError


def load_m5():
    raise NotImplementedError


def load_tourism1():
    raise NotImplementedError


def load_tourism2():
    raise NotImplementedError


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
    'M4'                     : load_m4,
    'M5'                     : load_m5,
    'UWAVE'                  : load_uwave,
    'Physionet 2012'         : load_physionet2012,
    'Physionet 2019'         : load_physionet2019,
    'USHCN'                  : load_ushcn,
    'tourism1'               : load_tourism1,
    'tourism2'               : load_tourism2,
}


def load_dataset(dataset: str):
    assert dataset_available(dataset)

    if not DATASETDIR.joinpath(dataset).exists():
        download_dataset(dataset)
        clean_dataset(dataset)

    return dataset_loaders[dataset]()
