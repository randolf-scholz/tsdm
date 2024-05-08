r"""Preprocessed subset of the USHCN climate dataset used by De Brouwer et al."""

__all__ = ["USHCN_DeBrouwer2019"]

import pandas as pd
from pandas import DataFrame

from tsdm.datasets.base import SingleTableDataset


class USHCN_DeBrouwer2019(SingleTableDataset):
    r"""Preprocessed subset of the USHCN climate dataset used by De Brouwer et al.

    References:
        - | `GRU-ODE-Bayes: Continuous Modeling of Sporadically﹣Observed Time Series
            <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>`_
          | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves
          | `Advances in Neural Information Processing Systems 2019
            <https://proceedings.neurips.cc/paper/2019>`_
    """

    SOURCE_URL = (
        r"https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/"
        r"master/gru_ode_bayes/datasets/Climate/"
    )
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = "https://github.com/edebrouwer/gru_ode_bayes"
    r"""HTTP address containing additional information about the dataset."""

    rawdata_files = ["small_chunked_sporadic.csv"]
    rawdata_hashes = {
        "small_chunked_sporadic.csv": "sha256:671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf",
    }
    rawdata_schemas = {
        "small_chunked_sporadic.csv": {
            "ID": "int16",
            "Time": "float32",
            "Value_0": "float32",
            "Value_1": "float32",
            "Value_2": "float32",
            "Value_3": "float32",
            "Value_4": "float32",
            "Mask_0": "bool",
            "Mask_1": "bool",
            "Mask_2": "bool",
            "Mask_3": "bool",
            "Mask_4": "bool",
        }
    }
    dataset_hash = (
        "sha256:bbd12ab38b4b7f9c69a07409c26967fe16af3b608daae9816312859199b5ce86"
    )

    def clean_table(self) -> DataFrame:
        r"""Clean an already downloaded raw dataset and stores it in hdf5 format."""
        fname = "small_chunked_sporadic.csv"
        file = self.rawdata_paths[fname]
        df = pd.read_csv(file, dtype=self.rawdata_schemas[fname])

        # replace missing values with NaN, using the mask
        channels = {}
        for k in range(5):
            key = f"CH_{k}"
            value = f"Value_{k}"
            channels[key] = value
            df[key] = df[value].where(df[f"Mask_{k}"])

        # set index and sort
        df = (
            df[["ID", "Time", *channels]]
            .set_index(["ID", "Time"])
            .sort_index()
            .rename(columns=channels)
        )

        return df
