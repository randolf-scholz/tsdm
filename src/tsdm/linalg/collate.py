r"""Pack, pad, unpack, and unpad batches of variable-length tensors."""

__all__ = [
    # Functions
    "collate_packed",
    "collate_padded",
    "unpad_sequence",
    "unpack_sequence",
]

from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from .logical_operators import cumulative_and


def collate_packed(batch: list[Tensor], /) -> PackedSequence:
    r"""Collate tensors into a `PackedSequence`, sorting the input list in place.

    The tensors must have positive lengths and matching trailing dimensions.
    The output is ordered by decreasing sequence length.
    """
    # First, need to sort in descending order by length
    batch.sort(key=Tensor.__len__, reverse=True)
    return pack_sequence(batch)


def collate_padded(
    batch: list[Tensor],  # Float[$N, ...]
    /,
    *,
    batch_first: bool = True,
    padding_value: float = 0.0,
) -> Tensor:  # Float[$Nₘₐₓ, ...]
    r"""Collate tensors of varying lengths into a single padded tensor.

    Equivalent to `torch.nn.utils.rnn.pad_sequence`, with `batch_first=True`
    by default. Trailing dimensions must match across tensors.

    Args:
        batch: The nonempty list of tensors to pad.
        batch_first: Whether to place the batch dimension before the time dimension.
        padding_value: The value used for padding; defaults to zero.
    """
    return pad_sequence(batch, batch_first=batch_first, padding_value=padding_value)


def unpack_sequence(batch: PackedSequence, /) -> list[Tensor]:
    r"""Unpack a `PackedSequence` into a list of unpadded tensors."""
    batch_pad_packed, lengths = pad_packed_sequence(batch, batch_first=True)
    return [x[:n] for x, n in zip(batch_pad_packed, lengths, strict=True)]


def unpad_sequence(
    padded_seq: Tensor,
    /,
    *,
    batch_first: bool = False,
    lengths: Optional[Tensor] = None,
    padding_value: float = torch.nan,
) -> list[Tensor]:
    r"""Recover unpadded tensors from a batch produced by `pad_sequence`.

    Args:
        padded_seq: A padded batch with matching trailing feature dimensions.
        batch_first: Whether the first dimension indexes the batch.
        lengths: The length of each sequence. If omitted, infer lengths by removing
            trailing timestamps whose features all equal `padding_value`.
        padding_value: The padding marker used when inferring lengths. NaN markers
            are detected with `torch.isnan`.

    Interior timestamps matching the padding marker are retained. Supply `lengths`
    when valid trailing observations can also match the padding marker.
    """
    # swap batch dimension if necessary
    padded_seq = padded_seq if batch_first else padded_seq.swapaxes(0, 1)  # (B, T, ...)

    # autodetect lengths if not provided
    if lengths is None:
        # convert padding value to scalar tensor
        padding: Tensor = torch.tensor(
            padding_value, dtype=padded_seq.dtype, device=padded_seq.device
        )

        # mask where tensor agrees with padding value
        mask = (
            torch.isnan(padded_seq) if torch.isnan(padding) else (padded_seq == padding)
        )

        # select the feature dimensions
        dims = tuple(range(min(2, padded_seq.ndim), padded_seq.ndim))

        # mask for completely missing timestamps and flip
        masked_timestamps = torch.all(mask, dim=dims)  # (B, T)
        # reverse along the time dimension
        masked_timestamps = masked_timestamps.flip(dims=(1,))  # (B, T)
        # cumulative aggregation of the mask
        masked_timestamps = cumulative_and(masked_timestamps, dim=1)  # (B, T)

        # count, starting from the back, until the first observation occurs.
        lengths = (~masked_timestamps).sum(dim=1)  # (B,)

    # FIXME: Why does pyright infer Unknown | Tensor | None?
    return [x[:n] for x, n in zip(padded_seq, lengths, strict=True)]
