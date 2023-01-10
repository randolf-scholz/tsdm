r"""LinODEnet pretrained Models."""

__all__ = [
    # Classes
    "OldLinODEnet",
    "LinODEnet",
]

import pickle

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Index, MultiIndex
from torch.nn.utils.rnn import pad_sequence

from tsdm.models.pretrained.base import PreTrainedModel

CHECKPOINTS = {
    "2022-11-16": "https://tubcloud.tu-berlin.de/s/ZQCatR7m28KBR3L/download/"
    "2022-11-16-linodenet-152669f30f5e5325bf916b154262eed5.zip",
    "2022-11-24": "https://tubcloud.tu-berlin.de/s/3ctPs6syJEHCJy4/download/"
    "2022-11-24-linodenet-a44fc91eab7a98130266d1c37f072eb5.zip",
    "2022-12-01": "https://tubcloud.tu-berlin.de/s/QgyJ58eW9JnZENz/download/"
    "2022-12-01-linodenet-50-f79c5e885f2182835e7b88ba3205fb33.zip",
}


class OldLinODEnet(PreTrainedModel):  # Deprecated
    """Import pre-trained LinODEnet model."""

    rawdata_file = "linodenet.zip"
    DOWNLOAD_URL = (
        "https://tubcloud.tu-berlin.de/s/syEZCZrBqQXiA5i/download/linodenet.zip"
    )
    RAWDATA_HASH = "15897965202b8e66db0189f4778655a3c55d350ca406447d8571133cbdfb1732"
    HASHES = {
        "model": ...,
        "encoder": ...,
        "optimizer": ...,
    }
    component_files = {
        "model": "LinODEnet-70",
        "encoder": "encoder.pickle",
    }


class LinODEnet(PreTrainedModel):
    r"""Import pre-trained LinODEnet model."""

    rawdata_file = "2022-11-16-linodenet-152669f30f5e5325bf916b154262eed5.zip"
    DOWNLOAD_URL = (
        f"https://tubcloud.tu-berlin.de/s/ZQCatR7m28KBR3L/download/{rawdata_file}"
    )
    RAWDATA_HASH = "d50d128b29e7310b4a9496494bea1ca1b614a7ffbf730f5a61d0b3026cb87ed8"

    component_files = {
        "model": "LinODEnet",
        "encoder": "encoder.pickle",
        "optimizer": "optimizer",
        "hyperparameters": "hparams.yaml",
    }

    def predict(self, ts: DataFrame) -> DataFrame:
        r"""Predict function for LinODEnet."""
        ts = self.clean_timeseries(ts)
        return self.get_predictions(ts)

    def clean_timeseries(self, ts: DataFrame) -> DataFrame:
        r"""Preprocess the time-series input."""
        USED_COLUMNS = Index(self.encoder[-1].column_encoders)  # type: ignore[index]

        columns = ts.columns
        used_columns = list(columns.intersection(USED_COLUMNS))
        drop_columns = list(columns.difference(USED_COLUMNS))
        miss_columns = list(USED_COLUMNS.difference(columns))

        # drop unused columns
        print(f">>> Dropping columns {drop_columns}")
        ts = ts.loc[:, used_columns]

        # fill up missing columns
        print(f">>> Adding columns {miss_columns}")
        ts.loc[:, miss_columns] = float("nan")

        # correctly order columns
        ts = ts[list(USED_COLUMNS)].copy()

        # fixing timestamp_type
        ts = ts.reset_index("measurement_time")
        if ts["measurement_time"].dtype != "timdedelta64":
            print(">>> Converting float (seconds) to timedelta64")
            ts["measurement_time"] = ts["measurement_time"] * np.timedelta64(1, "s")
        ts = ts.set_index(["measurement_time"], append=True)
        return ts

    @torch.no_grad()
    def get_predictions(self, ts: DataFrame) -> DataFrame:
        r"""Get predictions from the model."""
        if isinstance(ts.index, MultiIndex):
            names = ts.index.names[:-1]
            sizes = ts.groupby(names).size()
            T, X = self.encoder.encode(ts).values()
            T = T.to(device=self.device)
            X = X.to(device=self.device)
            T_list = torch.split(T, sizes.to_list())
            X_list = torch.split(X, sizes.to_list())
            T = pad_sequence(T_list, batch_first=True, padding_value=torch.nan)
            X = pad_sequence(X_list, batch_first=True, padding_value=torch.nan)

            XHAT = self.model(T, X)

            predictions = (
                {"T": t[:size], "X": xhat[:size]}
                for t, xhat, size in zip(T, XHAT, sizes)
            )
            d = {
                key: self.encoder.decode(pred)
                for key, pred in zip(sizes.index, predictions)
            }
            return pd.concat(d, names=names)

        # single time-series
        T, X = self.encoder.encode(ts).values()
        T = T.to(device=self.device)
        X = X.to(device=self.device)
        XHAT = self.model(T, X)
        decoded = self.encoder.decode({"T": T, "X": XHAT})
        return decoded

    @staticmethod
    def make_dataframes_from_pickle(
        filename: str,
    ) -> tuple[DataFrame, DataFrame, DataFrame]:
        r"""Returns DataFrames from pickle.

        Pickle must return a nested dictionary of the schema:

        .. code-block:: python

            KEYS = Literal["measurements_aggregated", "metadata", "setpoints"]
            dict[Any, dict[KEYS, DataFrame]]
        """
        with open(filename, "rb") as file:
            data = pickle.load(file)

        timeseries_dict = {
            key: tables["measurements_aggregated"] for key, tables in data.items()
        }
        timeseries = pd.concat(timeseries_dict, names=["experiment_id"])

        metadata_dict = {key: tables["metadata"] for key, tables in data.items()}
        metadata = pd.concat(metadata_dict, names=["experiment_id"])

        setpoints_dict = {key: tables["setpoints"] for key, tables in data.items()}
        setpoints = pd.concat(setpoints_dict, names=["experiment_id"])

        return timeseries, metadata, setpoints
