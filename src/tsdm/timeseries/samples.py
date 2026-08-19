r"""Utilities for time series samples."""

__all__ = [
    "SeparateTimeSample",
    "MergedTimeSample",
    "TripletSample",
]


from typing import Protocol, ReadOnly


class SeparateTimeSample[ArrayT](Protocol):
    r"""Protocol for forecasting requests.

    Args:
        context_times: Float[..., $N], padded NaN, non-decreasing
        context_values: Float[..., $N, D], padded NaN
        context_mask: Bool[..., $N, D], padded False
        query_times: Float[..., $K], padded NaN, non-decreasing
        query_mask: Bool[..., $K, F]  padded False
        target_values: Float[..., $K, F]  padded NaN
        static_covariates: Float[..., M]  padded NaN

    """

    context_times: ReadOnly[ArrayT]  # type: ignore
    context_values: ReadOnly[ArrayT]  # type: ignore
    context_mask: ReadOnly[ArrayT]  # type: ignore

    query_times: ReadOnly[ArrayT]  # type: ignore
    query_mask: ReadOnly[ArrayT]  # type: ignore
    target_values: ReadOnly[ArrayT | None] = None  # type: ignore

    static_covariates: ReadOnly[ArrayT | None] = None  # type: ignore


class MergedTimeSample[ArrayT](Protocol):
    r"""Protocol for joint time representation.

    Args:
        timestamps: Float[..., $T], padded NaN, non-decreasing
        context_mask: Bool[..., $T, D], padded False
        context_values: Float[..., $T, D], padded NaN
        query_mask: Bool[..., $T, E], padded False
        target_values: Float[..., $T, E], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    timestamps: ReadOnly[ArrayT]  # type: ignore

    context_mask: ReadOnly[ArrayT]  # type: ignore
    context_values: ReadOnly[ArrayT]  # type: ignore

    query_mask: ReadOnly[ArrayT]  # type: ignore
    target_values: ReadOnly[ArrayT | None] = None  # type: ignore

    static_covariates: ReadOnly[ArrayT | None] = None  # type: ignore


class TripletSample[ArrayT](Protocol):
    r"""Protocol for triplet representation.

    Tall data format that stacks context and query data into a 3 column representation of
    (time, channel, value) triplets.

    Args:
        context_times: Float[..., $X], padded NaN, non-decreasing
        context_channels: Long[..., $X], padded -1
        context_values: Float[..., $X], padded NaN
        query_times: Float[..., $Q], padded NaN, non-decreasing
        query_channels: Long[..., $Q], padded -1
        target_values: Float[..., $Q], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    context_times: ReadOnly[ArrayT]  # type: ignore
    context_channels: ReadOnly[ArrayT]  # type: ignore
    context_values: ReadOnly[ArrayT]  # type: ignore

    query_times: ReadOnly[ArrayT]  # type: ignore
    query_channels: ReadOnly[ArrayT]  # type: ignore
    target_values: ReadOnly[ArrayT | None] = None  # type: ignore

    static_covariates: ReadOnly[ArrayT | None] = None  # type: ignore
