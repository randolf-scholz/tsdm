r"""LinODEnet pretrained Models."""

__all__ = [
    # Classes
    "LinODEnet",
]

import pickle

import numpy as np
import pandas
import pandas as pd
import torch
from pandas import DataFrame, Index, MultiIndex
from torch.nn.utils.rnn import pad_sequence

from tsdm.models.pretrained.base import PreTrainedModel
from tsdm.utils.remote import download


class LinODEnet(PreTrainedModel):
    r"""Import pre-trained LinODEnet model."""

    DOCUMENTATION_URL = "https://bvt-htbd.gitlab-pages.tu-berlin.de/kiwi/tf1/linodenet/"
    CHECKPOINT_URL = "https://tubcloud.tu-berlin.de/s/P7SAkkaeGtAWJ2L?path=/LinODEnet"
    DOWNLOAD_URL = (
        "https://tubcloud.tu-berlin.de/s/P7SAkkaeGtAWJ2L/download?path=/LinODEnet/"
    )

    component_files = {
        "model": "LinODEnet",
        "encoder": "encoder.pickle",
        "optimizer": "optimizer",
        "hyperparameters": "hparams.yaml",
        "lr_scheduler": "lr_scheduler",
    }

    @classmethod
    def available_checkpoints(cls) -> DataFrame:
        download(
            cls.DOWNLOAD_URL + "checkpoints.xlsx", cls.RAWDATA_DIR / "checkpoints.xlsx"
        )
        return pandas.read_excel(cls.RAWDATA_DIR / "checkpoints.xlsx")

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
        r"""Return DataFrames from pickle.

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
