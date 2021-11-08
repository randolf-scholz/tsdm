r"""#TODO add module summary line.

#TODO add module description.
"""


import logging
from functools import cache, cached_property, partial
from itertools import product
from typing import Any, Callable, Literal, Optional

import torch
from pandas import DataFrame
from sklearn.model_selection import ShuffleSplit
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tsdm.datasets import KIWI_RUNS
from tsdm.datasets.base import DataSetCollection
from tsdm.encoders import ENCODERS, Encoder
from tsdm.encoders.functional import time2float
from tsdm.tasks.base import BaseTask
from tsdm.util import initialize_from
from tsdm.util.samplers import CollectionSampler, SequenceSampler

__logger__ = logging.getLogger(__name__)


class KIWI_RUNS_TASK(BaseTask):
    r"""A collection of bioreactor runs.

    For this task we do several simplifications

    - drop run_id 355
    -


    - timeseries for each run_id and experiment_id
    - metadata for each run_id and experiment_id

    When first do a train/test split.
    Then the goal is to learn a model in a multi-task fashion on all the ts.

    To train, we sample
    1. random TS from the dataset
    2. random snippets from the sampled TS

    Questions:
    - Should each batch contain only snippets form a single TS, or is there merit to sampling
      snippets from multiple TS in each batch?



    Divide 'Glucose' by 10, 'OD600' by 20, 'DOT' by 100, 'Base' by 200, then use RMSE.
    """

    keys: list[tuple[int, str]] = list(product(range(5), ("train", "test")))
    r"""Available keys."""
    KEYS = tuple[Literal[0, 1, 2, 3, 4], Literal["train", "test"]]
    r"""Type Hint for Keys"""
    timeseries: DataFrame
    r"""The whole timeseries data."""
    metadata: DataFrame
    r"""The metadata."""

    train_batch_size: int = 32
    """Default batch size."""
    eval_batch_size: int = 128
    """Default batch size when evaluating."""
    observation_horizon: int = 96
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: int = 24
    r"""The number of datapoints the model should forecast."""
    targets: list[str] = ["Glucose", "OD600", "DOT", "Base"]
    r"""The columns that should be predicted."""
    pre_encoder: Encoder
    r"""Encoder for the observations."""

    def test_metric(self) -> Callable[..., Tensor]:
        r"""The target metric."""
        ...

    def __init__(
        self,
        *,
        preprocessor: str = "StandardScaler",
        pre_processor: Optional[Encoder] = None,
        pre_encoder: Optional[Encoder] = None,
        forecasting_horizon: int = 24,
        observation_horizon: int = 96,
        eval_batch_size: int = 128,
        train_batch_size: int = 32,
    ):
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size

        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.horizon = self.observation_horizon + self.forecasting_horizon

        md: DataFrame = KIWI_RUNS.metadata
        ts: DataFrame = KIWI_RUNS.dataset

        # drop all time stamps outside of start_time and end_time in the metadata.
        merged = ts[[]].join(md[["start_time", "end_time"]])
        time = merged.index.get_level_values("measurement_time")
        cond = (merged["start_time"] <= time) & (time <= merged["end_time"])
        ts = ts[cond]
        # all measurements should be ≥0
        ts = ts.clip(lower=0.0)
        # remove run 355
        ts = ts.drop(355)
        md = md.drop(355)

        self.metadata: DataFrame = md
        self.timeseries: DataFrame = ts

        if pre_encoder is not None:
            self.preprocessor = initialize_from(ENCODERS, __name__=preprocessor)
        else:
            self.preprocessor = None

        if pre_encoder is not None:
            self.pre_encoder = initialize_from(ENCODERS, __name__=pre_encoder)
        else:
            self.pre_encoder = None

        self.loss_weights = {
            "Base": 200,
            "DOT": 100,
            "Glucose": 10,
            "OD600": 20,
        }

    @cached_property
    def split_idx(self) -> DataFrame:
        r"""Return table with indices for each split.

        Returns
        -------
        DataFrame
        """
        splitter = ShuffleSplit(n_splits=5, random_state=0, test_size=0.25)
        groups = self.metadata.groupby(["color", "run_id"])
        group_idx = groups.ngroup()

        splits = DataFrame(index=self.metadata.index)
        for i, (train, _) in enumerate(splitter.split(groups)):
            splits[i] = group_idx.isin(train).map({False: "test", True: "train"})

        splits.columns.name = "split"
        return splits.astype("string").astype("category")

    @cache
    def splits(self, key: KEYS) -> tuple[DataFrame, DataFrame]:
        """Return a subset of the data corresponding to the split.

        Parameters
        ----------
        key: tuple[int, str]

        Returns
        -------
        tuple[DataFrame, DataFrame]
        """
        assert key in self.keys, f"Wrong {key=}. Only {self.keys} work."
        split, data_part = key

        mask = self.split_idx[split] == data_part
        idx = self.split_idx[split][mask].index
        timeseries = self.timeseries.reset_index(level=2).loc[idx]
        timeseries = timeseries.set_index("measurement_time", append=True)
        metadata = self.metadata.loc[idx]
        return timeseries, metadata

    @cached_property
    def dataloaders(self) -> dict[tuple[int, str], DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        # TODO: make lazy dict
        return {
            key: self.get_dataloader(
                key, batch_size=self.eval_batch_size, shuffle=False, drop_last=False
            )
            for key in self.keys
        }

    @cached_property
    def batchloader(self) -> dict[int, DataLoader]:
        r"""Return the trainloader for the given key.

        Returns
        -------
        dict[int, DataLoader]
        """
        return {
            split: self.get_dataloader(
                (split, "train"),
                batch_size=self.train_batch_size,
                shuffle=True,
                drop_last=True,
            )
            for split in range(5)
        }

    def get_dataloader(
        self,
        key: tuple[int, str],
        batch_size: int = 1,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader for the training-dataset with the given batch_size.

        If encode=True, then it will create a dataloader with two outputs

        (inputs, targets)

        where inputs = pre_encoder.encode(masked_batch).

        Parameters
        ----------
        key: Literal["train", "valid", "test"]
            Dataset part from which to construct the DataLoader
        batch_size: int = 32
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
            defaults to cuda if cuda is available.
        shuffle: bool = True

        Returns
        -------
        DataLoader
        """
        split, part = key

        if part == "test":
            assert not shuffle, "Don't shuffle when evaluating test-dataset!"
        if part == "test" and "drop_last" in kwargs:
            assert not kwargs["drop_last"], "Don't drop when evaluating test-dataset!"

        ts, md = self.splits(key)
        ts_train, md_train = self.splits((split, "train"))

        # select metadata
        md = md[["Feed_concentration_glc", "pH_correction_factor", "OD_Dilution"]]
        # convert to regular float
        md = md.astype("float32")
        ts = ts.astype("float32")

        self.preprocessor.fit(ts_train)
        self.preprocessor.transform(ts, copy=False)

        ts = ts.reset_index(level=2)  # make measurements regular col

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        T = torch.tensor(
            time2float(ts["measurement_time"].values), device=device, dtype=dtype
        )
        X = torch.tensor(
            ts.drop(columns=["measurement_time"]).values, device=device, dtype=dtype
        )
        # M = torch.tensor(md, device=device, dtype=dtype)

        shared_index = md.index
        masks = {idx: (ts.index == idx) for idx in shared_index}
        datasets = {
            idx: TensorDataset(T[masks[idx]], X[masks[idx]]) for idx in shared_index
        }

        dataset = DataSetCollection(datasets)
        subsampler = partial(SequenceSampler, seq_len=self.horizon, shuffle=shuffle)  # type: ignore[assignment]
        sampler = CollectionSampler(dataset, subsampler=subsampler)

        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, **kwargs)
