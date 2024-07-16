r"""General Purpose Data Loaders for Time Series Data.

We implement multiple levels of abstraction.

- Dataloader for TimeSeriesTensor
- Dataloader for tuple of TimeSeriesTensor
- Dataloader for MetaDataset
   - sample dataset by index, then sample from that dataset.
"""

__all__ = [
    # Functions
    "collate_packed",
    "collate_padded",
    "unpad_sequence",
    "unpack_sequence",
]

import torch
from torch import Tensor
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from typing_extensions import Optional

from tsdm.linalg import cumulative_and


def collate_packed(batch: list[Tensor], /) -> PackedSequence:
    r"""Collates list of tensors into a PackedSequence."""
    # First, need to sort in descending order by length
    batch.sort(key=Tensor.__len__, reverse=True)
    return pack_sequence(batch)


def collate_padded(
    batch: list[Tensor], /, *, batch_first: bool = True, padding_value: float = 0.0
) -> Tensor:
    r"""Collates a list of tensors of varying lengths into a single Tensor, padded with zeros.

    Equivalent to `torch.nn.utils.rnn.pad_sequence`, but with `batch_first=True` as default

    .. signature:: ``[ (lᵢ, ...)_{i=1:B} ] -> (B, lₘₐₓ, ...)``.
    """
    return pad_sequence(batch, batch_first=batch_first, padding_value=padding_value)


def unpack_sequence(batch: PackedSequence, /) -> list[Tensor]:
    r"""Reverse operation of pack_sequence."""
    batch_pad_packed, lengths = pad_packed_sequence(batch, batch_first=True)
    return [x[:n] for x, n in zip(batch_pad_packed, lengths, strict=True)]


def unpad_sequence(
    padded_seq: Tensor,
    /,
    *,
    batch_first: bool = False,
    lengths: Optional[Tensor] = None,
    padding_value: float = float("nan"),
) -> list[Tensor]:
    r"""Reverse operation of `torch.nn.utils.rnn.pad_sequence`."""
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
        dims: list[int] = list(range(min(2, padded_seq.ndim), padded_seq.ndim))

        # mask for completely missing timestamps and flip
        masked_timestamps = torch.all(mask, dim=dims)  # (B, T)
        # reverse along the time dimension
        masked_timestamps = masked_timestamps.flip(dims=(1,))  # (B, T)
        # cumulative aggregation of the mask
        masked_timestamps = cumulative_and(masked_timestamps, dim=1)  # (B, T)

        # count, starting from the back, until the first observation occurs.
        lengths = (~masked_timestamps).sum(dim=1)  # (B,)

    # FIXME: Why does pyright infer Unknown | Tensor | None?
    return [x[:n] for x, n in zip(padded_seq, lengths, strict=True)]  # pyright: ignore[reportArgumentType]
