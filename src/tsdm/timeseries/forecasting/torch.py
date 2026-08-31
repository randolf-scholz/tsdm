r"""Torch-based utilities for time series forecasting."""

__all__ = [
    "SplitTimeData",
    "MergedTimeData",
    "TripletTimeData",
    "PaddedBatch",
    "collate_timeseries",
    # converters
    "split_to_merged",
    "triplet_to_split",
    "split_to_triplet",
    "triplet_to_merged",
    "merged_to_triplet",
    "merged_to_split",
]


from collections.abc import Iterable
from math import nan as NAN
from typing import NamedTuple, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tsdm.pprint import pprint_repr

from . import abstract

type AbstractSplitTimeSample = abstract.SplitTimeData[Tensor]
type AbstractMergedTimeSample = abstract.MergedTimeData[Tensor]
type AbstractTripletSample = abstract.TripletTimeData[Tensor]


@pprint_repr
class SplitTimeData(NamedTuple):
    r"""A single sample of time series data.

    Examples:
        time series forecasting/imputation::

            yhat = model(
                sample.query_times,
                sample.context_times,
                sample.context_values,
            )
            loss = d(sample.target_values, yhat)

    Attributes:
        context_times:     Float[..., $N], padded NaN, non-decreasing
        context_values:    Float[..., $N, D], padded NaN
        context_mask:      Bool[..., $N, D], padded False
        query_times:       Float[..., $K], padded NaN, non-decreasing
        query_mask:        Bool[..., $K, F],  padded False
        target_values:     Float[..., $K, F],  padded NaN
        static_covariates: Float[..., M],  padded NaN
    """

    context_times: Tensor
    context_mask: Tensor
    context_values: Tensor

    query_times: Tensor
    query_mask: Tensor

    target_values: Tensor | None = None
    static_covariates: Tensor | None = None

    @classmethod
    def from_unbatched(cls, samples: Iterable[SplitTimeData], /) -> SplitTimeData:
        raise NotImplementedError

    @classmethod
    def from_split(cls, sample: AbstractSplitTimeSample, /) -> SplitTimeData:
        return SplitTimeData(
            context_times=sample.context_times,
            context_mask=sample.context_mask,
            context_values=sample.context_values,
            query_times=sample.query_times,
            query_mask=sample.query_mask,
            target_values=sample.target_values,
            static_covariates=sample.static_covariates,
        )

    @classmethod
    def from_merged(cls, sample: AbstractMergedTimeSample, /) -> SplitTimeData:
        return merged_to_split(sample)

    @classmethod
    def from_triplet(cls, sample: AbstractTripletSample, /) -> SplitTimeData:
        return triplet_to_split(sample)

    def unbatch(self) -> list[SplitTimeData]:
        raise NotImplementedError

    def to_split(self) -> SplitTimeData:
        return self

    def to_merged(self) -> MergedTimeData:
        return split_to_merged(self)

    def to_triplet(self) -> TripletTimeData:
        return split_to_triplet(self)


@pprint_repr
class MergedTimeData(NamedTuple):
    r"""A single sample of time series data.

    Examples:
        time series forecasting/imputation::

            yhat = model(
                sample.timestamps,
                sample.context_values,
            )
            loss = d(sample.target_values, yhat)

    Attributes:
        timestamps:        Float[..., $T], padded NaN, non-decreasing
        context_mask:      Bool[..., $T, D], padded False
        context_values:    Float[..., $T, D], padded NaN
        query_mask:        Bool[..., $T, E], padded False
        target_values:     Float[..., $T, E], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    timestamps: Tensor
    context_mask: Tensor
    context_values: Tensor

    query_mask: Tensor

    target_values: Tensor | None = None
    static_covariates: Tensor | None = None

    @classmethod
    def from_unbatched(
        cls, samples: Iterable[AbstractMergedTimeSample], /
    ) -> MergedTimeData:
        raise NotImplementedError

    @classmethod
    def from_split(cls, sample: AbstractSplitTimeSample, /) -> MergedTimeData:
        return split_to_merged(sample)

    @classmethod
    def from_merged(cls, sample: AbstractMergedTimeSample, /) -> MergedTimeData:
        return MergedTimeData(
            timestamps=sample.timestamps,
            context_mask=sample.context_mask,
            context_values=sample.context_values,
            query_mask=sample.query_mask,
            target_values=sample.target_values,
            static_covariates=sample.static_covariates,
        )

    @classmethod
    def from_triplet(cls, sample: AbstractTripletSample, /) -> MergedTimeData:
        return triplet_to_merged(sample)

    def unbatch(self) -> list[MergedTimeData]:
        raise NotImplementedError

    def to_split(self) -> SplitTimeData:
        return merged_to_split(self)

    def to_merged(self) -> MergedTimeData:
        return self

    def to_triplet(self) -> TripletTimeData:
        return merged_to_triplet(self)


@pprint_repr
class TripletTimeData(NamedTuple):
    r"""A single sample of time series data.

    Examples:
        time series forecasting/imputation::

            yhat = model(
                sample.query_times,
                sample.context_times,
                sample.context_values,
            )
            loss = d(sample.target_values, yhat)

    Attributes:
        context_times:     Float[..., $X], padded NaN, non-decreasing
        context_channels:  Long[..., $X], padded -1
        context_values:    Float[..., $X], padded NaN
        query_times:       Float[..., $Q], padded NaN, non-decreasing
        query_channels:    Long[..., $Q], padded -1
        target_values:     Float[..., $Q], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    context_times: Tensor
    context_channels: Tensor
    context_values: Tensor

    query_times: Tensor
    query_channels: Tensor
    target_values: Tensor | None = None
    static_covariates: Tensor | None = None

    @classmethod
    def from_unbatched(
        cls, samples: Iterable[AbstractTripletSample], /
    ) -> TripletTimeData:
        raise NotImplementedError

    @classmethod
    def from_split(cls, sample: AbstractSplitTimeSample, /) -> TripletTimeData:
        return split_to_triplet(sample)

    @classmethod
    def from_merged(cls, sample: AbstractMergedTimeSample, /) -> TripletTimeData:
        return merged_to_triplet(sample)

    @classmethod
    def from_triplet(cls, sample: AbstractTripletSample, /) -> TripletTimeData:
        return TripletTimeData(
            context_times=sample.context_times,
            context_channels=sample.context_channels,
            context_values=sample.context_values,
            query_times=sample.query_times,
            query_channels=sample.query_channels,
            target_values=sample.target_values,
            static_covariates=sample.static_covariates,
        )

    def unbatch(self) -> list[TripletTimeData]:
        raise NotImplementedError

    def to_split(self) -> SplitTimeData:
        return triplet_to_split(self)

    def to_merged(self) -> MergedTimeData:
        return triplet_to_merged(self)

    def to_triplet(self) -> TripletTimeData:
        return self


def split_to_triplet(sample: AbstractSplitTimeSample) -> TripletTimeData:
    raise NotImplementedError


def split_to_merged(sample: AbstractSplitTimeSample) -> MergedTimeData:
    raise NotImplementedError


def merged_to_split(sample: AbstractMergedTimeSample) -> SplitTimeData:
    raise NotImplementedError


def merged_to_triplet(sample: AbstractMergedTimeSample) -> TripletTimeData:
    raise NotImplementedError


def triplet_to_split(sample: AbstractTripletSample) -> SplitTimeData:
    raise NotImplementedError


def triplet_to_merged(sample: AbstractTripletSample) -> MergedTimeData:
    raise NotImplementedError


@pprint_repr
class PaddedBatch(NamedTuple):
    r"""A padded batch of samples.

    Note:
        - B: batch-size
        - T: maximum sequence length (= N + K)
        - D: input dimension
        - K: target dimension

    Example:
        t, x, y, mq, mx, my = collate_timeseries(batch)
        yhat = model(t, x)  # predict for the whole padded batch
        r = where(my, y-yhat, 0)  # set residual to zero for missing values
        loss = r.abs().pow(2).sum()  # compute the loss

    Attributes:
        t (Float[B×T]):   Combined timestamps of inputs and targets.
        x (Float[B×T×D]): The inputs for all timesteps.
        y (Float[B×T×F]): The targets for the target timesteps.
        mq (Bool[B×T]):   The 'queries' mask, True if the given time stamp is a query.
        mx (Bool[B×T×D]): The 'inputs' mask, True indicates an observation, False a missing value.
        my (Bool[B×T×F]): The 'targets' mask, True indicates a target, False a missing value.
        static_covariates (Optional[Float[B×M]]): Stacked static covariates.

    In this context, 'query' means that the model should predict the value at this time stamp.
    Consequently, `mq` is identical to or-reducing `my` along the target dimension.
    """

    t: Tensor  # B×N:   the padded timestamps/queries.
    x: Tensor  # B×N×D: the padded input values.
    y: Tensor  # B×N×F: the padded target values.
    mq: Tensor  # B×N:   the 'queries' mask.
    mx: Tensor  # B×N×D: the 'inputs' mask.
    my: Tensor  # B×N×F: the 'targets' mask.
    static_covariates: Optional[Tensor] = None  # B×M: stacked covariates.


def collate_timeseries(batch: list[SplitTimeData]) -> PaddedBatch:
    r"""Collate timeseries samples into padded batch.

    Assumptions:
        - `query_times` is sorted.
    """
    masks_inputs: list[Tensor] = []
    masks_queries: list[Tensor] = []
    masks_target: list[Tensor] = []
    padded_inputs: list[Tensor] = []
    padded_queries: list[Tensor] = []
    padded_targets: list[Tensor] = []
    static_covariates: list[Tensor] = []

    for sample in batch:
        if sample.static_covariates is not None:
            static_covariates.append(sample.static_covariates)

        t_inputs = sample.context_times
        x = sample.context_values
        t_target = sample.query_times
        y = sample.target_values
        if y is None:
            raise ValueError("Cannot collate a forecasting sample without targets.")

        # pad the x-values by the target length
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]),
            fill_value=NAN,
            device=x.device,
            dtype=x.dtype,
        )
        x_padded = torch.cat((x, x_padding))

        # get the whole time interval
        t_combined = torch.cat((t_inputs, t_target))
        sorted_idx = torch.argsort(t_combined)

        # create a mask for looking up the target values
        m_queries = torch.cat(
            [
                torch.zeros_like(t_inputs, dtype=torch.bool),
                sample.query_mask.any(dim=-1),
            ]
        )
        input_mask_padding = torch.zeros(
            (t_target.shape[0], x.shape[-1]),
            dtype=torch.bool,
            device=x.device,
        )
        m_inputs = torch.cat((sample.context_mask, input_mask_padding))
        m_targets = sample.query_mask

        # append to lists, ordering by time
        masks_inputs.append(m_inputs[sorted_idx])
        masks_queries.append(m_queries[sorted_idx])
        masks_target.append(m_targets)  # assuming t_target is sorted
        padded_inputs.append(x_padded[sorted_idx])
        padded_queries.append(t_combined[sorted_idx])
        padded_targets.append(y)  # assuming t_target is sorted

    return PaddedBatch(
        t=pad_sequence(padded_queries, batch_first=True).squeeze(),
        x=pad_sequence(padded_inputs, batch_first=True, padding_value=NAN).squeeze(),
        y=pad_sequence(padded_targets, batch_first=True, padding_value=NAN).squeeze(),
        mq=pad_sequence(masks_queries, batch_first=True).squeeze(),
        mx=pad_sequence(masks_inputs, batch_first=True).squeeze(),
        my=pad_sequence(masks_target, batch_first=True).squeeze(),
        static_covariates=(
            torch.stack(static_covariates) if static_covariates else None
        ),
    )
