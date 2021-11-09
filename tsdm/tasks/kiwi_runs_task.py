r"""#TODO add module summary line.

#TODO add module description.
"""


import logging
from functools import cache, cached_property, partial
from itertools import product
from typing import Any, Callable, Literal, Optional

import torch
from pandas import DataFrame, Series
from sklearn.model_selection import ShuffleSplit
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tsdm.datasets import KIWI_RUNS, Dataset
from tsdm.datasets.base import DataSetCollection
from tsdm.encoders.modular import (
    BaseEncoder,
    ChainedEncoder,
    DataFrameEncoder,
    DateTimeEncoder,
    FloatEncoder,
    Standardizer,
    TensorEncoder,
)
from tsdm.tasks.base import BaseTask
from tsdm.util.samplers import CollectionSampler, SequenceSampler

__logger__ = logging.getLogger(__name__)


class KIWI_RUNS_TASK(BaseTask):
    r"""A collection of bioreactor runs.

    For this task we do several simplifications

    - drop run_id 355
    - drop almost all metadata
    - restrict timepoints to start_time & end_time given in metadata.


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
    dataset: Dataset = KIWI_RUNS
    r"""The Dataset."""

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
    pre_encoder: BaseEncoder
    r"""Encoder for the observations."""
    controls: Series
    r"""The control variables."""
    targets: Series
    r"""The target varaibles."""
    observables: Series
    r"""The observables."""
    loss_weights: DataFrame
    r"""Contains Information about the loss."""

    @cached_property
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The target metric."""
        ...

    def __init__(
        self,
        *,
        preprocessor: Optional[BaseEncoder] = None,
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

        if preprocessor is None:
            self.preprocessor = ChainedEncoder(
                TensorEncoder(),
                DataFrameEncoder(
                    Standardizer() @ FloatEncoder(),
                    index_encoder=DateTimeEncoder(),
                ),
            )

        controls = Series(
            [
                "Cumulated_feed_volume_glucose",
                "Cumulated_feed_volume_medium",
                "InducerConcentration",
                "StirringSpeed",
                "Flow_Air",
                "Temperature",
                "Probe_Volume",
            ]
        )
        controls.index = controls.apply(lambda x: ts.columns.get_loc(x))
        self.controls = controls

        targets = Series(
            [
                "Base",
                "DOT",
                "Glucose",
                "OD600",
            ]
        )
        targets.index = targets.apply(lambda x: ts.columns.get_loc(x))
        self.targets = targets

        observables = Series(
            [
                "Acetate",
                "Fluo_GFP",
                "Volume",
                "pH",
            ]
        )
        observables.index = observables.apply(lambda x: ts.columns.get_loc(x))
        self.observables = observables

        assert (set(controls) | set(targets) | set(observables)) == set(ts.columns)

        # Setting of loss weights
        weights = DataFrame.from_dict(
            {
                "Base": 200,
                "DOT": 100,
                "Glucose": 10,
                "OD600": 20,
            },
            orient="index",
            columns=["weight"],
        )
        weights["col_index"] = weights.index.map(lambda x: (ts.columns == x).argmax())
        weights.index.name = "col"
        self.loss_weights = weights

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
    def splits(self, key: KEYS) -> tuple[DataFrame, DataFrame]:  # type: ignore[override]
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
    def batchloader(self) -> dict[int, DataLoader]:  # type: ignore[override]
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
        # select metadata
        md = md[["Feed_concentration_glc", "pH_correction_factor", "OD_Dilution"]]
        # convert to regular float
        md = md.astype("float32")
        ts = ts.astype("float32")

        # preprocessing
        ts_train, md_train = self.splits((split, "train"))
        self.preprocessor.fit(ts_train.reset_index(level=[0, 1], drop=True))
        T, X = self.preprocessor.transform(ts.reset_index(level=[0, 1], drop=True))

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dtype = torch.float32
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
