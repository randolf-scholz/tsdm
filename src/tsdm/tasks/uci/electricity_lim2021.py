r"""Code for the electricity task from the paper "Temporal Fusion Transformers" by Lim et al. (2021)."""

__all__ = [
    # CLASSES
    "ElectricityLim2021",
    "Batch",
]


from collections.abc import Callable
from functools import cached_property
from typing import Any, Literal, NamedTuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, default_collate

from tsdm.datatools import timedelta, timestamp
from tsdm.encoders import StandardScaler
from tsdm.pprint import pprint_repr
from tsdm.random.samplers import Sampler, SlidingWindowSampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import TimeSeries, electricity


@pprint_repr
class Batch(NamedTuple):
    r"""A batch of input and target tensors with timestamps and masks."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


type SplitID = Literal["train", "test", "valid", "joint", "whole"]


class ElectricityLim2021(
    TimeSeriesTask[SplitID, TimeSeries, Any, tuple[Tensor, ...], tuple[Tensor, ...]]
):
    r"""Experiments as performed by the "TFT" paper.

    References:
        - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting <br>
          Lim et al. <br>
          International Journal of Forecasting 2021 <br>
          <https://www.sciencedirect.com/science/article/pii/S0169207021000637>

    ### Implementation notes

    - Hourly observations are aggregated with the mean. The TRMF description says
        that blocks of four quarter-hourly readings are aggregated but does not
        specify the operation in the quoted passage. Since readings are in kW,
        energy consumption in kWh requires accounting for the sampling interval.
    - Training ends at `2014-08-08`; validation ends at `2014-09-01`. These are
        chronological partitions rather than a random 90%/10% split of windows.
    - The preprocessing encoder is fitted on the training partition.
    - Validation and test partitions include the preceding seven days as context.
        Test targets span the seven days beginning on `2014-09-01`; the sampler
        advances by one day with a seven-day observation window and a one-day
        forecasting window.
    - The error function computes normalized absolute error on the tensors passed
        to it. Callers must decode predictions and targets to evaluate on the
        original scale.

    ### Evaluation protocol

    > In accordance with [9], we use the past week (i.e. 168 hours) to
    > forecast over the next 24 hours.
    >
    > Electricity: Per [9], we use 500k samples taken between 2014-01-01 to 2014-09-01 – using
    > the first 90% for training, and the last 10% as a validation set. Testing is done over the
    > 7 days immediately following the training set – as described in [9, 32]. Given the large
    > differences in magnitude between trajectories, we also apply z-score normalization
    > separately to each entity for real-valued inputs. In line with previous work, we consider
    > the electricity usage, day-of-week, hour-of-day and a time index – i.e. the number of
    > time steps from the first observation – as real-valued inputs, and treat the entity
    > identifier as a categorical variable.

    ### Test metrics

    Evaluation uses quantile risk at $q=0.5$ and $q=0.9$ (P50 and P90):

    $$ q-\text{Risk} = 2\frac{∑_{yₜ} ∑_{τ} QL(y(t), ŷ(t-τ), q)}{∑_y ∑_{τ} |y(t)|} $$

    ### Results

    | Model | ARIMA | ConvTrans | DSSM  | DeepAR | ETS   | MQRNN | Seq2Seq | TFT   | TRMF  |
    |-------|-------|-----------|-------|--------|-------|-------|---------|-------|-------|
    | P50   | 0.154 | 0.059     | 0.083 | 0.075  | 0.102 | 0.077 | 0.067   | 0.055 | 0.084 |
    | P90   | 0.102 | 0.034     | 0.056 | 0.400  | 0.077 | 0.036 | 0.036   | 0.027 | NaN   |
    """

    def __init__(self) -> None:
        ds = electricity().timeseries
        ds = ds.resample("1h").mean()
        mask = (self.boundaries["start"] <= ds.index) & (
            ds.index < self.boundaries["final"]
        )
        dataset = TimeSeries("electricity", timeseries=ds[mask])

        self.observation_period = timedelta("7d")
        self.forecasting_period = timedelta("1d")
        self.observation_horizon = 24 * 7
        self.forecasting_horizon = 24

        super().__init__(dataset=dataset, error_fn=self.q50_loss)
        self.encoder = StandardScaler(axis=-1)
        self.encoder.fit(self.partitions["train"].timeseries)

    @property
    def split_keys(self) -> tuple[SplitID, ...]:
        return tuple(self.masks)

    def is_train_split(self, key: SplitID) -> bool:
        r"""Return whether the split is `"train"` or the combined `"joint"` partition."""
        return key in {"train", "joint"}

    @cached_property
    def boundaries(self) -> dict[str, pd.Timestamp]:
        r"""Start and end dates of the training, validation and test sets."""
        return {
            "start": timestamp("2014-01-01"),
            "train": timestamp("2014-08-08"),
            "valid": timestamp("2014-09-01"),
            "final": timestamp("2014-09-08"),
        }

    @cached_property
    def masks(self) -> dict[SplitID, pd.Series]:
        r"""Masks for the training, validation and test sets."""
        return {
            "train": (self.boundaries["start"] <= self.dataset.timeindex)
            & (self.dataset.timeindex < self.boundaries["train"]),
            "valid": (
                self.boundaries["train"] - self.observation_period
                <= self.dataset.timeindex
            )
            & (self.dataset.timeindex < self.boundaries["valid"]),
            "test": (
                self.boundaries["valid"] - self.observation_period
                <= self.dataset.timeindex
            )
            & (self.dataset.timeindex < self.boundaries["final"]),
            "whole": (self.boundaries["start"] <= self.dataset.timeindex)
            & (self.dataset.timeindex < self.boundaries["final"]),
            "joint": (self.boundaries["start"] <= self.dataset.timeindex)
            & (self.dataset.timeindex < self.boundaries["valid"]),
        }

    def make_partition(self, key: SplitID, /) -> TimeSeries:
        r"""Return the time series restricted to the specified partition."""
        return self.dataset[self.masks[key].to_list()]

    def make_generator(self, key: SplitID, /) -> TensorDataset:
        r"""Create the encoded tensor dataset for the specified split."""
        encoded = self.encoder.encode(self.partitions[key].timeseries)
        tensor = torch.tensor(encoded.values, dtype=torch.float32)
        return TensorDataset(tensor)

    def make_sampler(self, key: SplitID, /) -> Sampler:
        r"""Sample complete observation and prediction windows."""
        return SlidingWindowSampler(
            self.partitions[key].timeindex,
            stride="1d",
            horizons=self.observation_period + self.forecasting_period,
            mode="index",
            drop_last=True,
            shuffle=self.is_train_split(key),
        )

    def make_collate_fn(
        self, _key: SplitID, /
    ) -> Callable[[list[tuple[Tensor, ...]]], tuple[Tensor, ...]]:
        r"""Stack equally sized tensor windows into a batch."""
        return lambda samples: tuple(default_collate(samples))

    @staticmethod
    def q50_loss(predictions: Tensor, targets: Tensor, /) -> Tensor:
        r"""Normalized absolute error, equivalent to the q=0.5 quantile risk."""
        return torch.sum(torch.abs(predictions - targets)) / torch.sum(
            torch.abs(targets)
        )
