r"""Utilities for time series samples."""

__all__ = [
    "SeparateTimeSample",
    "MergedTimeSample",
    "TripletSample",
]


from typing import Protocol, ReadOnly


class SeparateTimeSample[ArrayT](Protocol):
    r"""Protocol for forecasting requests."""

    context_times: ReadOnly[ArrayT]  # Float[..., $N], padded NaN, non-decreasing
    context_values: ReadOnly[ArrayT]  # Float[..., $N, D], padded NaN
    context_mask: ReadOnly[ArrayT]  # Bool[..., $N, D], padded False

    query_times: ReadOnly[ArrayT]  # Float[..., $K], padded NaN, non-decreasing
    query_mask: ReadOnly[ArrayT]  # Bool[..., $K, F]  padded False
    target_values: ReadOnly[ArrayT | None] = None  # Float[..., $K, F]  padded NaN

    static_covariates: ReadOnly[ArrayT | None] = None  # Float[..., M]  padded NaN


class MergedTimeSample[ArrayT](Protocol):
    r"""Protocol for joint time representation."""

    timestamps: ReadOnly[ArrayT]  # Float[..., $T], padded NaN, non-decreasing

    context_mask: ReadOnly[ArrayT]  # Bool[..., $T, D], padded False
    context_values: ReadOnly[ArrayT]  # Float[..., $T, D], padded NaN

    query_mask: ReadOnly[ArrayT]  # Bool[..., $T, E], padded False
    target_values: ReadOnly[ArrayT | None] = None  # Float[..., $T, E], padded NaN

    static_covariates: ReadOnly[ArrayT | None] = None  # Float[..., M], padded NaN


class TripletSample[ArrayT](Protocol):
    r"""Protocol for triplet representation.

    Tall data format that stacks context and query data into a 3 column representation of
    (time, channel, value) triplets.
    """

    context_times: ReadOnly[ArrayT]  # Float[..., $X], padded NaN, non-decreasing
    context_channels: ReadOnly[ArrayT]  # Long[..., $X], padded -1
    context_values: ReadOnly[ArrayT]  # Float[..., $X], padded NaN

    query_times: ReadOnly[ArrayT]  # Float[..., $Q], padded NaN, non-decreasing
    query_channels: ReadOnly[ArrayT]  # Long[..., $Q], padded -1
    target_values: ReadOnly[ArrayT | None] = None  # Float[..., $Q], padded NaN

    static_covariates: ReadOnly[ArrayT | None] = None  # Float[..., M], padded NaN
