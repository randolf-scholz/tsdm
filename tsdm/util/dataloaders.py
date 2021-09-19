r"""General Purpose Data Loaders for Time Series Data.

We implement multiple levels of abstraction.

- Dataloader for TimeSeriesTensor
- Dataloader for tuple of TimeSeriesTensor
- Dataloader for MetaDataset
   - sample dataset by index, then sample from that dataset.

"""

import logging
from typing import Callable, Final, Iterator, Optional, Sequence, Sized, TypeVar, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from torch.utils.data import Sampler

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "collate_list",
    "collate_packed",
    "collate_padded",
    "unpad_sequence",
    "upack_sequence",
    "SliceSampler",
    "TimeSliceSampler",
]


T = TypeVar("T")
Sample = Union[T, Callable[[], T]]


def collate_list(batch: list[Tensor]) -> list[Tensor]:
    r"""Collates list of tensors as list of tensors."""
    return batch


def collate_packed(batch: list[Tensor]) -> PackedSequence:
    r"""Collates list of tensors into a PackedSequence."""
    # First, need to sort in descending order by length
    batch.sort(key=Tensor.__len__, reverse=True)
    return pack_sequence(batch)


def collate_padded(
    batch: list[Tensor], batch_first: bool = True, padding_value: float = 0.0
) -> Tensor:
    r"""Collates list of tensors of varying lengths into a single Tensor, padded with zeros.

    Equivalent to :func:`torch.nn.utils.rnn.pad_sequence`, but with `batch_first=True` as default

    Signature: `[ (l_i, ...)_{i=1:B} ] -> (B, l_{\max},...)`

    Parameters
    ----------
    batch: list[Tensor]
    batch_first: bool, default=True
    padding_value: float, default=True

    Returns
    -------
    Tensor
    """
    return pad_sequence(batch, batch_first=batch_first, padding_value=padding_value)


def upack_sequence(batch: PackedSequence) -> list[Tensor]:
    r"""Reverse operation of pack_sequence."""
    batch_pad_packed, lengths = pad_packed_sequence(batch, batch_first=True)
    torch.swapaxes(batch_pad_packed, 1, 2)
    return [x[:l].T for x, l in zip(batch_pad_packed, lengths)]


def unpad_sequence():
    r"""Reverse operation of pad_sequence."""
    print(help(pack_padded_sequence))


class TimeSliceSampler(Sampler):
    """TODO: add class."""

    def __init__(self, data_source: Optional[Sized]):
        """TODO: Add method."""
        super().__init__(data_source)

    def __iter__(self) -> Iterator:
        """TODO: Add method."""
        return super().__iter__()


# TODO: add exclusive_args decorator
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
        slice_sampler: Optional[Sample[int]] = None,
        sampler: Optional[Callable[[], tuple[int, int]]] = None,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = np.arange(len(data_source))
        self.rng = np.random.default_rng()

        def slicesampler_dispatch() -> Callable[[], int]:
            if slice_sampler is None:
                return lambda: max(1, len(data_source) // 10)
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
