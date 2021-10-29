r"""KIWI Run Data

TODO: Module Summary
"""

from __future__ import annotations

__all__ = [
    # Classes
    "KIWI_RUNS",
]

import pickle
import logging
from pathlib import Path
from functools import cache
from collections import defaultdict

from pandas import DataFrame, Series
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from itertools import chain

from tsdm.datasets.dataset import BaseDataset

LOGGER = logging.getLogger(__name__)


def contains_no_information(series: Series) -> bool:
    return len(series.dropna().unique()) <= 1


def contains_nan_slice(series: Series, slices: list[Series]) -> bool:
    for idx in slices:
        if pd.isna(series[idx]).all():
            return True
    return False


def create_replicate_dict(experiments_per_run):
    """Stores the list of possible (run_id, experiment_id) for each
    replicate set as given by a tuple (run_id, color) in a dictionary

    args:

    experiment_per_run:  dict of dict of dict as given for the present dataset.
                         keys of first level: run_ids
                         keys of second level: experiment_ids
                         keys of third level: metadata, measurements_reactor
                                              measurements_array, setpoints,
                                              measurements_aggregated


    returns: dict  (maps (run_id, experiment_id) to the list of (run_id, experiment_id) that belongs to it.)

    """

    col_run_to_exp = defaultdict(list)
    for run in experiments_per_run.keys():
        for exp in experiments_per_run[run].keys():
            col_run_to_exp[
                (experiments_per_run[run][exp]["metadata"]["color"][0], run)
            ].append((run, exp))
    return col_run_to_exp


class ReplicateBasedSplitter:
    def __init__(self, n_splits=5, random_state=0, test_size=0.25, train_size=None):
        self.splitter = ShuffleSplit(
            n_splits=n_splits,
            random_state=random_state,
            test_size=test_size,
            train_size=train_size,
        )  #

    def split(self, col_run_to_exp):
        """generator that yields the lists of  pairs of keys to create the train and test data.
        Example usage s. below"""
        keys = list(col_run_to_exp.keys())
        for train_repl_sets, test_repl_sets in self.splitter.split(keys):
            train_keys = list(
                chain(
                    *[col_run_to_exp[keys[key_index]] for key_index in train_repl_sets]
                )
            )
            test_keys = list(
                chain(
                    *[col_run_to_exp[keys[key_index]] for key_index in test_repl_sets]
                )
            )
            yield train_keys, test_keys


metadata_dtypes = {
    "experiment_id": "UInt32",
    "bioreactor_id": "UInt32",
    "container_number": "UInt32",
    "profile_id": "UInt32",
    "color": "string",
    "profile_name": "string",
    "organism_id": "UInt32",
    "run_id": "UInt32",
    "OD_Dilution": "float32",
    "run_name": "string",
    "start_time": "datetime64[ns]",
    "end_time": "datetime64[ns]",
    "Feed_concentration_glc": "float32",
    "pH_correction_factor": "float32",
}

metadata_categoricals = {
    "profile_name": "category",
    "run_name": "category",
    "color": "category",
    "pH_correction_factor": "Float32",
    "Feed_concentration_glc": "Float32",
    "OD_Dilution": "Float32",
}


class KIWI_RUNS(BaseDataset):
    r"""KIWI RUN Data.

    .. code-block:: python

        list[
            dict[
                "train": dict[(int, int), dict[
                    'metadata': DataFrame,                  # MetaData
                    'setpoints': DataFrame,                 # MetaData
                    'measurements_reactor': DataFrame,      # TimeTensor
                    'measurements_array': DataFrame,        # TimeTensor
                    'measurements_aggregated' : DataFrame,  # TimeTensor
                ],
                "test": dict[(int, int), dict[
                    'metadata': DataFrame,                  # MetaData
                    'setpoints': DataFrame,                 # MetaData
                    'measurements_reactor': DataFrame,      # TimeTensor
                    'measurements_array': DataFrame,        # TimeTensor
                    'measurements_aggregated' : DataFrame,  # TimeTensor
                ],
            ]
        ]
    """

    url: str = (
        "https://owncloud.innocampus.tu-berlin.de/index.php/s/"
        "fRBSr82NxY7ratK/download/kiwi_experiments_and_run_355.pk"
    )

    @classmethod
    @property
    @cache
    def rawdata_file(cls) -> Path:
        path = cls.rawdata_path.joinpath("kiwi_experiments_and_run_355.pk")
        path.mkdir(parents=True, exist_ok=True)

    @classmethod  # type: ignore[misc]
    @property
    @cache
    def dataset_file(cls) -> Path:
        r"""Path of the dataset file."""
        return cls.dataset_path.joinpath(f"{cls.__name__}.pk")  # type: ignore[attr-defined]

    @classmethod
    def clean(cls):
        """Create `DataFrame` with 1 column per client and `DatetimeIndex`."""
        dataset = cls.__name__
        LOGGER.info("Cleaning dataset '%s'", dataset)

        with open(cls.rawdata_file, "rb") as file:
            data = pickle.load(file)

        DATA = [
            (data[run][exp] | {"run_id": run, "experiment_id": exp})
            for run in data
            for exp in data[run]
        ]
        DF = DataFrame(DATA).set_index(["run_id", "experiment_id"])
        metadata = pd.concat(iter(DF["metadata"]), keys=DF["metadata"].index)
        setpoints = pd.concat(iter(DF["setpoints"]), keys=DF["setpoints"].index)
        measurements_reactor = pd.concat(
            iter(DF["measurements_reactor"]), keys=DF["measurements_reactor"].index
        )
        measurements_array = pd.concat(
            iter(DF["measurements_array"]), keys=DF["measurements_array"].index
        )
        measurements_aggregated = pd.concat(
            iter(DF["measurements_aggregated"]),
            keys=DF["measurements_aggregated"].index,
        )

        # with open(cls.dataset_file, "w") as file:
        #     pickle.dump(cv_splits, file)

        LOGGER.info("Finished cleaning dataset '%s'", dataset)

    @classmethod
    def load(cls):
        r"""Load the dataset from hdf-5 file."""
        super().load()  # <- makes sure DS is downloaded and preprocessed
        with open(cls.dataset_file, "rb") as file:
            data = pickle.load(file)
        return data
