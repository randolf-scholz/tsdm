r"""Implementation of the kiwi task."""

__all__ = [
    # Classes
    "Batch",
    "KiwiBenchmark",
]


from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, Literal, NamedTuple

import torch
from pandas import DataFrame
from torch import Tensor, nan as NAN
from torch.nn.utils.rnn import pad_sequence

from tsdm.constants import EMPTY_MAP, RNG
from tsdm.datatools import (
    CallableDataset,
    folds_as_frame,
    folds_as_sparse_frame,
    folds_from_groups,
)
from tsdm.encoders import (
    BoundaryEncoder,
    BoxCoxEncoder,
    DateTimeEncoder,
    Encoder,
    LogitBoxCoxEncoder,
    MinMaxScaler,
    StandardScaler,
)
from tsdm.encoders.pandas import FrameAsTensorDict, FrameEncoder
from tsdm.losses.sequence import mse_loss
from tsdm.pprint import pprint_repr
from tsdm.random.samplers import HierarchicalSampler, Sampler, SlidingWindowSampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import (
    SplitTimeData,
    TimeSeriesCollection,
    kiwi_benchmark,
    make_sample_factory,
)
from tsdm.utils import LazyDict


@pprint_repr
class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


type SplitID = tuple[int, Literal["train", "test", "valid"]]


class KiwiBenchmark(
    TimeSeriesTask[
        SplitID, TimeSeriesCollection[tuple[int, int]], Any, SplitTimeData, Batch
    ]
):
    r"""Task for the KIWI dataset.

    The task is to forecast the observables inside the forecasting horizon.
    """

    dataset: TimeSeriesCollection[tuple[int, int]]

    # sampler kwargs
    observation_horizon: str = "2h"
    r"""The interval of observational data."""
    forecasting_horizon: str = "1h"
    r"""The interval for which the model should forecast."""
    stride: str = "15min"
    r"""The stride of the sliding window sampler."""

    observables: list[str] = [
        "Base",
        "DOT",
        "Glucose",
        "OD600",
        "Acetate",
        "Fluo_GFP",
        "pH",
        "Temperature",
    ]
    r"""The channels to observe."""
    targets: list[str] = [
        "Base",
        "DOT",
        "Glucose",
        "OD600",
        "Acetate",
        "Fluo_GFP",
        "pH",
        "Temperature",
    ]
    r"""The channels to forecast."""
    covariates: list[str] = [
        "Cumulated_feed_volume_glucose",
        "Cumulated_feed_volume_medium",
        "InducerConcentration",
        "StirringSpeed",
        "Flow_Air",
        "Probe_Volume",
    ]
    r"""The covariates, which are known in advance."""
    fold_kwargs: dict[str, Any] = {
        "seed": 2022,
        "num_folds": 5,
        "train": 7,
        "valid": 1,
        "test": 2,
    }
    r"""The configuration of the fold generator."""
    generator_kwargs: dict[str, Any] = {
        "observables": observables,
        "covariates": covariates,
        "targets": targets,
    }
    r"""The configuration of the sample generator."""
    sampler_kwargs: dict[str, Any] = {
        "observation_horizon": observation_horizon,
        "forecasting_horizon": forecasting_horizon,
        "stride": stride,
        "early_stop": False,
        "shuffle": False,
    }
    r"""The configuration of the sampler."""
    dataloader_kwargs: dict[str, Any] = {
        "num_workers": 0,
        "pin_memory": True,
    }
    r"""The configuration of the dataloader."""

    def __init__(
        self,
        *,
        # observation_horizon: str = "2h",
        # forecasting_horizon: str = "1h",
        fold_kwargs: Mapping[str, Any] = EMPTY_MAP,
        sampler_kwargs: Mapping[str, Any] = EMPTY_MAP,
        generator_kwargs: Mapping[str, Any] = EMPTY_MAP,
        dataloader_kwargs: Mapping[str, Any] = EMPTY_MAP,
        train_batch_size: int = 32,
        eval_batch_size: int = 128,
    ) -> None:
        r"""Initialize the KIWI task."""
        self.generator_kwargs = self.generator_kwargs | dict(generator_kwargs)
        self.sampler_kwargs = self.sampler_kwargs | dict(sampler_kwargs)
        self.fold_kwargs = self.fold_kwargs | dict(fold_kwargs)
        self.dataloader_kwargs = self.dataloader_kwargs | dict(dataloader_kwargs)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.observables = self.generator_kwargs["observables"]
        self.covariates = self.generator_kwargs["covariates"]
        self.targets = self.generator_kwargs["targets"]
        self.stride = self.sampler_kwargs["stride"]
        self.observation_horizon = self.sampler_kwargs["observation_horizon"]
        self.forecasting_horizon = self.sampler_kwargs["forecasting_horizon"]

        super().__init__(
            dataset=kiwi_benchmark(),
            error_fn=lambda predictions, targets: mse_loss(
                predictions=predictions, targets=targets, time_dim=-2, channel_dim=-1
            ),
        )
        self.encoders = LazyDict.from_func(self.make_encoder, self.split_keys)

    @property
    def split_keys(self) -> tuple[SplitID, ...]:
        return tuple(self.split_masks.columns)

    def is_train_split(self, key: SplitID) -> bool:
        return key[-1] == "train"

    def make_partition(self, key: SplitID, /) -> TimeSeriesCollection[tuple[int, int]]:
        r"""Select complete trajectories using both components of their identifiers."""
        keys = self.split_masks.index[self.split_masks[key].to_numpy(dtype=bool)]
        metadata = self.dataset.static_covariates
        return TimeSeriesCollection(
            self.dataset.name,
            timeseries=self.dataset.timeseries.loc[
                self.dataset.timeindex.index.isin(keys)
            ],
            timeseries_metadata=self.dataset.timeseries_metadata,
            static_covariates=None if metadata is None else metadata.loc[keys],
            static_covariates_metadata=self.dataset.static_covariates_metadata,
            constants=self.dataset.constants,
            constants_metadata=self.dataset.constants_metadata,
        )

    @cached_property
    def split_masks(self) -> DataFrame:
        r"""Group by RunID and color which indicates replicates."""
        md = self.dataset.static_covariates
        if md is None:
            raise ValueError("KIWI fold generation requires static covariates.")
        groups = md.groupby(["run_id", "color"], sort=False).ngroup()
        folds = folds_from_groups(groups, **self.fold_kwargs)
        df = folds_as_frame(folds)
        return folds_as_sparse_frame(df)

    def make_collate_fn(
        self, key: SplitID, /
    ) -> Callable[[list[SplitTimeData]], Batch]:
        r"""Create the collate function for the given split."""
        encoder = self.encoders[key]

        def collate_fn(samples: list[SplitTimeData], /) -> Batch:
            x_vals: list[Tensor] = []
            y_vals: list[Tensor] = []
            x_time: list[Tensor] = []
            y_time: list[Tensor] = []
            x_mask: list[Tensor] = []
            y_mask: list[Tensor] = []

            for sample in samples:
                if sample.target_values is None:
                    raise ValueError(
                        "Cannot collate a forecasting sample without targets."
                    )
                tx, x = encoder.encode(sample.context_values).values()
                ty, y = encoder.encode(sample.target_values).values()
                # create a mask for looking up the target values
                x_time.append(tx)
                x_vals.append(x)
                x_mask.append(x.isfinite())

                y_time.append(ty)
                y_vals.append(y)
                y_mask.append(y.isfinite())

            return Batch(
                x_time=pad_sequence(x_time, batch_first=True).squeeze(),
                x_vals=pad_sequence(
                    x_vals, batch_first=True, padding_value=NAN
                ).squeeze(),
                x_mask=pad_sequence(x_mask, batch_first=True).squeeze(),
                y_time=pad_sequence(y_time, batch_first=True).squeeze(),
                y_vals=pad_sequence(
                    y_vals, batch_first=True, padding_value=NAN
                ).squeeze(),
                y_mask=pad_sequence(y_mask, batch_first=True).squeeze(),
            )

        return collate_fn

    def get_columns_encoder(self, variable: str, /) -> Encoder:
        descr = self.dataset.timeseries_metadata
        if descr is None:
            raise ValueError("KIWI encoders require timeseries metadata.")
        kind, lower, upper = descr.loc[variable, ["kind", "lower_bound", "upper_bound"]]

        match kind:
            case "percent" | "fraction":
                return (
                    StandardScaler()
                    @ LogitBoxCoxEncoder()
                    @ MinMaxScaler(xmin=lower, xmax=upper)
                    @ BoundaryEncoder(lower, upper, mode="clip")
                )
            case "absolute":
                return (
                    StandardScaler()
                    @ BoxCoxEncoder()
                    @ BoundaryEncoder(lower, upper, mode="clip")
                )
            case "linear":
                return StandardScaler()
            case _:
                raise ValueError(f"{kind=} unknown")

    def make_encoder(self, key: SplitID, /) -> Encoder:
        encoder = FrameEncoder(
            {
                col: self.get_columns_encoder(col)
                for col in self.dataset.timeseries.columns
            },
            elapsed_time=DateTimeEncoder() >> MinMaxScaler(),
        ) >> FrameAsTensorDict(
            schema={
                "T": ["elapsed_time"],
                "X": ...,
            },
            dtypes={"T": torch.float32, "X": torch.float32},
        )

        self.LOGGER.info("Initializing Encoder for key='%s'", key)
        train_key: SplitID = (key[0], "train")
        associated_train_split = self.partitions[train_key]
        self.LOGGER.info("Fitting encoder to associated train split '%s'", train_key)
        encoder.fit(
            associated_train_split.timeseries.droplevel(["run_id", "experiment_id"])
        )

        return encoder

    def make_sampler(self, key: SplitID, /, **kwds: Any) -> Sampler:
        split = self.partitions[key]

        # get configuration
        sampler_kwargs: dict[str, Any] = (
            self.sampler_kwargs | {"shuffle": self.is_train_split(key)} | kwds
        )
        observation_horizon = sampler_kwargs.pop("observation_horizon")
        forecasting_horizon = sampler_kwargs.pop("forecasting_horizon")
        stride = sampler_kwargs.pop("stride")
        early_stop = sampler_kwargs.pop("early_stop")
        shuffle = sampler_kwargs.pop("shuffle")
        rng = sampler_kwargs.pop("rng", RNG)

        if sampler_kwargs:
            raise ValueError(f"Unknown sampler_kwargs: {sampler_kwargs}")

        subsamplers = {
            key: SlidingWindowSampler(
                tsd.timeindex,
                horizons=[observation_horizon, forecasting_horizon],
                stride=stride,
                shuffle=shuffle,
                mode="mask",
                rng=rng,
            )
            for key, tsd in split.items()
        }

        sampler = HierarchicalSampler[tuple[int, int], Any](
            split,
            subsamplers,
            early_stop=early_stop,
            shuffle=shuffle,
            rng=rng,
        )

        return sampler

    def make_generator(
        self, key: SplitID, /, **kwds: Any
    ) -> CallableDataset[Any, SplitTimeData]:
        r"""Sample generator for the KIWI dataset."""
        # get configuration
        generator_kwargs = self.generator_kwargs | kwds
        observables = generator_kwargs.pop("observables")
        targets = generator_kwargs.pop("targets")
        covariates = generator_kwargs.pop("covariates")

        if generator_kwargs:
            raise ValueError(f"Unknown generator_kwargs: {generator_kwargs}")

        return make_sample_factory(
            self.partitions[key],
            observables=observables,
            targets=targets,
            covariates=covariates,
        )
