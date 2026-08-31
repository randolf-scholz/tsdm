r"""Utilities for time series samples."""

__all__ = [
    "SplitTimeData",
    "MergedTimeData",
    "TripletTimeData",
]


from typing import Protocol


class SplitTimeData[ArrayT](Protocol):
    r"""Protocol for forecasting requests.

    Attributes:
        context_times:     Float[..., $N], padded NaN, non-decreasing
        context_values:    Float[..., $N, D], padded NaN
        context_mask:      Bool[..., $N, D], padded False
        query_times:       Float[..., $K], padded NaN, non-decreasing
        query_mask:        Bool[..., $K, F],  padded False
        target_values:     Float[..., $K, F],  padded NaN
        static_covariates: Float[..., M],  padded NaN
    """

    # TODO: Use typing.ReadOnly (PEP 767)

    @property
    def context_times(self) -> ArrayT: ...
    @property
    def context_values(self) -> ArrayT: ...
    @property
    def context_mask(self) -> ArrayT: ...

    @property
    def query_times(self) -> ArrayT: ...
    @property
    def query_mask(self) -> ArrayT: ...
    @property
    def target_values(self) -> ArrayT | None: ...

    @property
    def static_covariates(self) -> ArrayT | None: ...


class MergedTimeData[ArrayT](Protocol):
    r"""Protocol for joint time representation.

    Attributes:
        timestamps:        Float[..., $T], padded NaN, non-decreasing
        context_mask:      Bool[..., $T, D], padded False
        context_values:    Float[..., $T, D], padded NaN
        query_mask:        Bool[..., $T, E], padded False
        target_values:     Float[..., $T, E], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    # TODO: Use typing.ReadOnly (PEP 767)

    @property
    def timestamps(self) -> ArrayT: ...

    @property
    def context_mask(self) -> ArrayT: ...
    @property
    def context_values(self) -> ArrayT: ...

    @property
    def query_mask(self) -> ArrayT: ...
    @property
    def target_values(self) -> ArrayT | None: ...

    @property
    def static_covariates(self) -> ArrayT | None: ...


class TripletTimeData[ArrayT](Protocol):
    r"""Protocol for triplet representation.

    Tall data format that stacks context and query data into a 3 column representation of
    (time, channel, value) triplets.

    Attributes:
        context_times:     Float[..., $X], padded NaN, non-decreasing
        context_channels:  Long[..., $X], padded -1
        context_values:    Float[..., $X], padded NaN
        query_times:       Float[..., $Q], padded NaN, non-decreasing
        query_channels:    Long[..., $Q], padded -1
        target_values:     Float[..., $Q], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    # TODO: Use typing.ReadOnly (PEP 767)

    @property
    def context_times(self) -> ArrayT: ...
    @property
    def context_channels(self) -> ArrayT: ...
    @property
    def context_values(self) -> ArrayT: ...

    @property
    def query_times(self) -> ArrayT: ...
    @property
    def query_channels(self) -> ArrayT: ...
    @property
    def target_values(self) -> ArrayT | None: ...

    @property
    def static_covariates(self) -> ArrayT | None: ...
