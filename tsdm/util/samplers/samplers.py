r"""Module Summary Line.

Module description
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

import logging
from typing import Callable, Final, Iterator, Optional, Sequence, Sized, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Sampler

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "SliceSampler",
    "TimeSliceSampler",
    "SequenceSampler",
]


class TimeSliceSampler(Sampler):
    """TODO: add class."""

    def __init__(self, data_source: Optional[Sized]):
        """TODO: Add method."""
        super().__init__(data_source)

    def __iter__(self) -> Iterator:
        """TODO: Add method."""
        return super().__iter__()


class SliceSampler(Sampler):
    r"""Sample by index.

    Default modus operandi:

    - Use fixed window size
    - Sample starting index uniformly from [0:-window]

    Should you want to sample windows of varying size, you may supply a

    Attributes
    ----------
    data:
    idx: range(len(data))
    rng: a numpy random Generator
    """

    data: Sequence
    idx: NDArray
    rng: np.random.Generator

    def __init__(
        self,
        data_source: Sequence,
        slice_sampler: Optional[Union[int, Callable[[], int]]] = None,
        sampler: Optional[Callable[[], tuple[int, int]]] = None,
        generator: Optional[np.random.Generator] = None,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = np.arange(len(data_source))
        self.rng = np.random.default_rng() if generator is None else generator

        def slicesampler_dispatch() -> Callable[[], int]:
            # use default if None is provided
            if slice_sampler is None:
                return lambda: max(1, len(data_source) // 10)
            # convert int to constant function
            if isinstance(slice_sampler, int):
                return lambda: slice_sampler  # type: ignore
            if callable(slice_sampler):
                return slice_sampler
            raise NotImplementedError("slice_sampler not compatible.")

        self._slice_sampler = slicesampler_dispatch()

        def default_sampler() -> tuple[int, int]:
            window_size: int = self._slice_sampler()
            start_index: int = self.rng.choice(self.idx[:-window_size])
            return window_size, start_index

        self._sampler = default_sampler if sampler is None else sampler

    def slice_sampler(self) -> int:
        r"""Return random window size."""
        return self._slice_sampler()

    def sampler(self) -> tuple[int, int]:
        r"""Return random start_index and window_size."""
        return self._sampler()

    def __iter__(self) -> Iterator:
        r"""Yield random slice from dataset.

        Returns
        -------
        Iterator
        """
        while True:
            # sample len and index
            window_size, start_index = self.sampler()
            # return slice
            yield self.data[start_index : start_index + window_size]


class SequenceSampler(Sampler):
    """Samples sequences of length seq_len."""

    def __init__(self, data_source, seq_len):
        super().__init__(data_source)
        self.data = data_source
        self.seq_len = seq_len

    def __iter__(self):
        """Return Indices of the Samples."""
        for idx in range(len(self.data) - self.seq_len):
            yield range(idx, idx + self.seq_len)
