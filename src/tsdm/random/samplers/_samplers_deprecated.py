r"""Deprecated samplers."""

__all__ = ["SequenceSampler"]

from collections.abc import Iterable, Iterator
from typing import Optional, cast
from warnings import deprecated

import numpy as np
from numpy.typing import NDArray

from tsdm.constants import RNG
from tsdm.random.samplers.base import BaseSampler
from tsdm.types.mixins import SupportsLenAndGetItem
from tsdm.types.scalars import DurationScalar, TimestampScalar
from tsdm.utils import timedelta, timestamp


@deprecated("Use SlidingWindowSampler instead.")
class SequenceSampler[TD: DurationScalar](BaseSampler):
    r"""Samples sequences of fixed length."""

    data: NDArray[TimestampScalar[TD]]  # type: ignore[type-var]
    seq_len: TD
    r"""The length of the sequences."""
    stride: TD
    r"""The stride at which to sample."""
    xmax: TimestampScalar[TD]
    r"""The maximum value at which to stop sampling."""
    xmin: TimestampScalar[TD]
    r"""The minimum value at which to start sampling."""
    return_mask: bool = False
    r"""Whether to return masks instead of indices."""
    shuffle: bool = False
    r"""Whether to shuffle the data."""

    def __init__(
        self,
        data_source: Iterable[TimestampScalar[TD]]
        | SupportsLenAndGetItem[TimestampScalar[TD]],
        /,
        *,
        return_mask: bool = False,
        seq_len: str | TD,
        shuffle: bool = False,
        stride: str | TD,
        tmin: Optional[str | TimestampScalar[TD]] = None,
        tmax: Optional[str | TimestampScalar[TD]] = None,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.data = np.asarray(data_source)

        match tmin:
            case None:
                self.xmin = self.data[0]
            case str(time_str):
                self.xmin = timestamp(time_str)
            case _:
                self.xmin = tmin

        match tmax:
            case None:
                self.xmax = self.data[-1]
            case str(time_str):
                self.xmax = timestamp(time_str)
            case _:
                self.xmax = tmax

        total_delta = self.xmax - self.xmin
        self.stride = cast(TD, timedelta(stride) if isinstance(stride, str) else stride)
        self.seq_len = cast(
            TD, timedelta(seq_len) if isinstance(seq_len, str) else seq_len
        )

        # k_max = max {k∈ℕ ∣ x_min + seq_len + k⋅stride ≤ x_max}
        self.k_max = int((total_delta - self.seq_len) // self.stride)
        self.return_mask = return_mask

        self.samples = np.array([
            (
                (x <= self.data) & (self.data < y)  # type: ignore[operator]
                if self.return_mask
                else [x, y]
            )
            for x, y in self._iter_tuples()
        ])

    def _iter_tuples(self) -> Iterator[tuple[TimestampScalar[TD], TimestampScalar[TD]]]:
        x = self.xmin
        y = x + self.seq_len
        # allows nice handling of negative seq_len
        x, y = min(x, y), max(x, y)  # type: ignore[call-overload]
        yield x, y

        for _ in range(len(self)):
            x = x + self.stride
            y = y + self.stride
            yield x, y

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return self.k_max

    def __iter__(self) -> Iterator:
        r"""Return an iterator over the samples."""
        n = len(self)
        index = RNG.permutation(n) if self.shuffle else np.arange(n)
        return iter(self.samples[index])

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}[{self.stride}, {self.seq_len}]"
