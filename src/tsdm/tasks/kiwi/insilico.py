r"""Forecasting task for the in silico bioreactor dataset."""

__all__ = ["InSilicoTask"]

from functools import cached_property
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, MultiIndex

from tsdm.losses.sequence import mse_loss
from tsdm.random.samplers import HierarchicalSampler, Sampler, SlidingWindowSampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.tasks.partitions import rotating_partitions
from tsdm.timeseries.pandas import (
    SplitTimeData,
    TimeSeriesCollection,
    in_silico,
    make_sample_factory,
)
from tsdm.types.datasets import CallableDataset

type SplitID = tuple[int, Literal["train", "test", "valid"]]
type SampleID = tuple[int, list[NDArray[np.bool_]]]


class InSilicoTask(
    TimeSeriesTask[SplitID, TimeSeriesCollection[int], SampleID, SplitTimeData]
):
    r"""Forecast biomass and product in simulated bioreactor trajectories."""

    observation_horizon: str = "2h"
    r"""Duration of the observation window, expressed as a pandas timedelta string."""
    forecasting_horizon: str = "1h"
    r"""Duration of the forecasting window, expressed as a pandas timedelta string."""

    def __init__(
        self,
        observation_horizon: str = "2h",
        forecasting_horizon: str = "1h",
    ) -> None:
        dataset = in_silico()
        self.observation_horizon = observation_horizon
        self.forecasting_horizon = forecasting_horizon
        super().__init__(
            dataset,
            error_fn=lambda predictions, targets: mse_loss(
                predictions=predictions, targets=targets, time_dim=-2, channel_dim=-1
            ),
        )

    @property
    def split_keys(self) -> tuple[SplitID, ...]:
        return tuple(self.split_masks.columns)

    def is_train_split(self, key: SplitID) -> bool:
        return key[-1] == "train"

    def make_partition(self, key: SplitID, /) -> TimeSeriesCollection[int]:
        return self.dataset[self.split_masks[key].to_list()]

    def make_sampler(self, key: SplitID, /) -> Sampler[SampleID]:
        split = self.partitions[key]
        subsamplers = {
            key: SlidingWindowSampler(
                tsd.timeindex,
                horizons=[self.observation_horizon, self.forecasting_horizon],
                stride="1h",
                mode="mask",
            )
            for key, tsd in split.items()
        }
        sampler = HierarchicalSampler[int, Any](
            split,
            subsamplers,
            shuffle=self.is_train_split(key),
        )
        return sampler

    @cached_property
    def split_masks(self) -> DataFrame:
        r"""Boolean masks indexed by series identifier and split key."""
        index = self.dataset.metaindex
        folds = rotating_partitions(
            index, num=8, chunks={"train": 6, "valid": 1, "test": 1}, rng=2022
        )
        masks = {
            (fold_id, split): index.isin(members)
            for fold_id, fold in enumerate(folds)
            for split, members in sorted(fold.items())
            if members
        }
        columns = MultiIndex.from_tuples(list(masks), names=["fold", "partition"])
        return DataFrame(masks, index=index, columns=columns)

    def make_generator(
        self, key: SplitID, /
    ) -> CallableDataset[SampleID, SplitTimeData]:
        split = self.partitions[key]
        return make_sample_factory(
            split,
            targets=["Biomass", "Product"],
            observables=["Biomass", "Substrate", "Acetate", "DOTm"],
            covariates=["Volume", "Feed"],
        )
