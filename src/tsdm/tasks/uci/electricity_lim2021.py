r"""Code for the electricty task from the paper "Temporal Fusion Transformer" by Lim et al. (2021)."""

__all__ = [
    # CLASSES
    "ElectricityLim2021",
    "Batch",
]


from functools import cached_property

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Any, Literal, NamedTuple

from tsdm.data.timeseries import TimeSeriesDataset
from tsdm.datasets import Electricity
from tsdm.encoders import BaseEncoder, StandardScaler
from tsdm.random.samplers import SequenceSampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.utils import timedelta, timestamp
from tsdm.utils.pprint import pprint_repr


@pprint_repr
class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


class ElectricityLim2021(TimeSeriesTask):
    r"""Experiments as performed by the "TFT" paper.

    Note that there is an issue: in the pipe-line, the hourly aggregation is done via mean,
    whereas in the TRMF paper, the hourly aggregation is done via plain summation.

    .. epigraph:: We convert the data to reflect hourly consumption, by aggregating blocks of 4 columns

    Issues:

    - They report in the paper: 90% train, 10% validation. However, this is wrong.
      They split the array not on the % of samples, but instead they use the first 218 days
      as train and the following 23 days ("2014-08-08" ≤ t < "2014-09-01" ) as validation,
      leading to a split of 90.12% train and 9.88% validation.
    - preprocessing: what partitions of the dataset are mean and variance computed over?
      train? train+validation?
    - Since the values are in kW, an hourly value would correspond to summing the values.
    - Testing: How exactly is the loss computed? From the description, it cannot be
      precisely inferred. Looking inside the code reveals that the test split is actually the
      last 14 days, which makes sense given that the observation period is 7 days. However,
      the paper does not mention the stride. Given the description:
      "we use the past week (i.e. 168 hours) to forecast over the next 24 hours."
      Does not tell the stride, i.e. how much the sliding window is moved. We assume this to be
      24h.
    - Is the loss computed on the original scale, or on the pre-processed (i.e. z-score normalized)
      scale? The code reveals that apparently the loss is computed on the original scale!

    Paper
    -----
    - | Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
      | https://www.sciencedirect.com/science/article/pii/S0169207021000637

    Evaluation Protocol
    -------------------
    .. epigraph::

        In accordance with [9], we use the past week (i.e. 168 hours) to
        forecast over the next 24 hours.

        Electricity: Per [9], we use 500k samples taken between 2014-01-01 to 2014-09-01 – using
        the first 90% for training, and the last 10% as a validation set. Testing is done over the
        7 days immediately following the training set – as described in [9, 32]. Given the large
        differences in magnitude between trajectories, we also apply z-score normalization
        separately to each entity for real-valued inputs. In line with previous work, we consider
        the electricity usage, day-of-week, hour-of-day and a time index – i.e. the number of
        time steps from the first observation – as real-valued inputs, and treat the entity
        identifier as a categorical variable.

    Test-Metric
    -----------
    Evaluation: $q$-Risk ($q=50$ and $q=90$)

    .. math:: q-Risk = 2\frac{∑_{y_t} ∑_{τ} QL(y(t), ŷ(t-τ), q)}{∑_y ∑_{τ} |y(t)|}

    Results
    -------
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | Model | ARIMA | ConvTrans | DSSM  | DeepAR | ETS   | MQRNN | Seq2Seq | TFT   | TRMF  |
    +=======+=======+===========+=======+========+=======+=======+=========+=======+=======+
    | P50   | 0.154 | 0.059     | 0.083 | 0.075  | 0.102 | 0.077 | 0.067   | 0.055 | 0.084 |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | P90   | 0.102 | 0.034     | 0.056 | 0.400  | 0.077 | 0.036 | 0.036   | 0.027 | NaN   |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    """

    KeyType = Literal["train", "test", "valid", "joint", "whole"]
    r"""Type Hint for index."""

    preprocessor: BaseEncoder

    # FIXME: need a different base class for this task!
    dataset: TimeSeriesDataset  # type: ignore[assignment]

    def __init__(self) -> None:
        ds = Electricity().table
        ds = ds.resample("1h").mean()
        mask = (self.boundaries["start"] <= ds.index) & (
            ds.index < self.boundaries["final"]
        )
        dataset = ds[mask]

        super().__init__(dataset=dataset)

        self.observation_period = timedelta("7d")
        self.forecasting_period = timedelta("1d")
        self.observation_horizon = 24 * 7
        self.forecasting_horizon = 24

        self.encoder = StandardScaler()
        self.encoder.fit(self.splits["train"])

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
    def masks(self) -> dict[KeyType, np.ndarray]:
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

    def make_split(self, key: KeyType) -> Any:
        r"""Return the split of the dataset."""
        return self.dataset[self.masks[key]]

    def make_dataloader(
        self, key: KeyType, /, *, shuffle: bool = False, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return the dataloader for the given key."""
        ds = self.splits[key]
        encoded = self.encoder.encode(ds)
        tensor = torch.tensor(encoded.values, dtype=torch.float32)

        sampler = SequenceSampler(
            encoded.index,
            stride="1d",
            seq_len=self.observation_period + self.forecasting_period,
            return_mask=True,
            shuffle=shuffle,
        )
        dataset = TensorDataset(tensor)

        return DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
