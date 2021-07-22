r"""General Purpose Data Loaders for Time Series Data."""
import logging

import torch
from torch import Tensor
from torch.nn.utils.rnn import (
    pack_sequence,
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    PackedSequence,
)

logger = logging.getLogger(__name__)
__all__ = [
    "collate_list",
    "collate_packed",
    "collate_padded",
    "unpad_sequence",
    "upack_sequence",
]


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

    Signature: $[ (l_i, ...)_{i=1:B} ] -> (B, l_{\max},...)$

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
    torch.swapaxes(batch_pad_packed, 1, 2), lengths
    return [x[:l].T for x, l in zip(batch_pad_packed, lengths)]


def unpad_sequence():
    r"""Reverse operation of pad_sequence."""
