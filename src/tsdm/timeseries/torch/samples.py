r"""PyTorch tensor structures, conversions, and batching for prediction samples.

This module implements the abstract split, merged, and triplet sample layouts
with tensors and collates variable-length split samples into padded batches.
Conversion functions between layouts are placeholders that raise
`NotImplementedError`.
"""

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
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tsdm.timeseries import abstract
from tsdm.utils import pretty_repr

type AbstractSplitTimeSample = abstract.SplitTimeData[Tensor]
type AbstractMergedTimeSample = abstract.MergedTimeData[Tensor]
type AbstractTripletSample = abstract.TripletTimeData[Tensor]


@pretty_repr
class SplitTimeData(NamedTuple):
    r"""A prediction sample with separate context and query time axes.

    Leading dimensions represent optional batch axes. A `$` prefix marks a
    dynamic dimension that may vary between samples. `$N` and `$K` are the
    context and query lengths; `D` and `F` are their channel counts, and `M`
    is the number of static covariates. Within each sample, timestamps are
    non-decreasing. Padded values and timestamps use `NaN`; masks use `False`.

    Attributes:
        context_times: Timestamps of type `Float[..., $N]`.
        context_values: Context values of type `Float[..., $N, D]`.
        context_mask: Availability mask of type `Bool[..., $N, D]`.
        query_times: Query timestamps of type `Float[..., $K]`.
        query_mask: Target mask of type `Bool[..., $K, F]`.
        target_values: Targets of type `Float[..., $K, F]`, or `None`.
        static_covariates: Covariates of type `Float[..., M]`, or `None`.

    Example:
        ```python
        yhat = model(
            sample.query_times,
            sample.context_times,
            sample.context_values,
        )
        loss = d(sample.target_values, yhat)
        ```
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


@pretty_repr
class MergedTimeData(NamedTuple):
    r"""A prediction sample with a shared context and query time axis.

    Leading dimensions represent optional batch axes. A `$` prefix marks a
    dynamic dimension that may vary between samples. `$T` is the shared
    sequence length, `D` and `E` are the context and target channel counts,
    and `M` is the number of static covariates. Within each sample, timestamps
    are non-decreasing. Padded values and timestamps use `NaN`; masks use `False`.

    Attributes:
        timestamps: Timestamps of type `Float[..., $T]`.
        context_mask: Availability mask of type `Bool[..., $T, D]`.
        context_values: Context values of type `Float[..., $T, D]`.
        query_mask: Target mask of type `Bool[..., $T, E]`.
        target_values: Targets of type `Float[..., $T, E]`, or `None`.
        static_covariates: Covariates of type `Float[..., M]`, or `None`.

    Example:
        ```python
        yhat = model(sample.timestamps, sample.context_values)
        loss = d(sample.target_values, yhat)
        ```
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


@pretty_repr
class TripletTimeData(NamedTuple):
    r"""A prediction sample represented by time, channel, and value triplets.

    Leading dimensions represent optional batch axes. A `$` prefix marks a
    dynamic dimension that may vary between samples. `$X` and `$Q` are the
    numbers of context and query triplets, and `M` is the number of static
    covariates. Within each sample, timestamps are non-decreasing. Padded
    values and timestamps use `NaN`; channel indices use `-1`.

    Attributes:
        context_times: Timestamps of type `Float[..., $X]`.
        context_channels: Channel indices of type `Long[..., $X]`.
        context_values: Context values of type `Float[..., $X]`.
        query_times: Query timestamps of type `Float[..., $Q]`.
        query_channels: Channel indices of type `Long[..., $Q]`.
        target_values: Targets of type `Float[..., $Q]`, or `None`.
        static_covariates: Covariates of type `Float[..., M]`, or `None`.

    Example:
        ```python
        yhat = model(
            sample.query_times,
            sample.query_channels,
            sample.context_times,
            sample.context_channels,
            sample.context_values,
        )
        loss = d(sample.target_values, yhat)
        ```
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


@pretty_repr
class PaddedBatch(NamedTuple):
    r"""A padded batch of samples.

    `B` is the batch size, `$T` the maximum combined context and query length,
    `$K` the maximum query length, `D` the input channel count, `F` the target
    channel count, and `M` the number of static covariates. A `$` prefix marks
    a dynamic dimension that may vary between batches. Shapes below are
    shown before `collate_timeseries` removes singleton dimensions with
    `squeeze()`. Values are padded with `NaN`, timestamps with zero, and
    masks with `False`.

    Attributes:
        t: Combined input and query timestamps of type `Float[B, $T]`.
        x: Input values on the combined time axis, of type `Float[B, $T, D]`.
        y: Target values on the query time axis, of type `Float[B, $K, F]`.
        mq: Mask of type `Bool[B, $T]`, true at query entries with targets.
        mx: Input availability mask of type `Bool[B, $T, D]`.
        my: Target availability mask of type `Bool[B, $K, F]`.
        static_covariates: Stacked covariates of type `Float[B, M]`, or `None`.

    The input and query axes remain separate for targets: `mq` marks query
    entries on the combined axis, while `my` marks values on the query axis.

    Example: Compute a masked loss from predictions on the query axis
        ```python
        padded = collate_timeseries(batch)
        yhat = model(padded)  # return predictions aligned with padded.y
        residual = torch.where(padded.my, padded.y - yhat, 0)
        loss = residual.abs().pow(2).sum()
        ```
    """

    t: Tensor  # B×N:   the padded timestamps/queries.
    x: Tensor  # B×N×D: the padded input values.
    y: Tensor  # B×N×F: the padded target values.
    mq: Tensor  # B×N:   the 'queries' mask.
    mx: Tensor  # B×N×D: the 'inputs' mask.
    my: Tensor  # B×N×F: the 'targets' mask.
    static_covariates: Tensor | None = None  # B×M: stacked covariates.


def collate_timeseries(batch: list[SplitTimeData]) -> PaddedBatch:
    r"""Collate unbatched split samples into a padded batch.

    Context and query timestamps are concatenated and sorted for each sample.
    Overlapping timestamps remain separate entries. Target values and masks
    retain the query time axis, so `query_times` must already be sorted.
    All samples must have compatible channel dimensions. If static covariates
    are supplied, provide them for every sample to retain batch alignment.

    Args:
        batch: A non-empty list of unbatched samples with target values.

    Returns:
        Padded tensors with singleton dimensions removed by `squeeze()`.
        Static covariates are stacked without squeezing, or are `None` when
        absent from every sample.

    Raises:
        ValueError: If a sample has no target values.
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
