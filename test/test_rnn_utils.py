#!usr/bin/env python
r"""Test RNN utils."""

import torch
from pytest import mark
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from tsdm.util.dataloaders import unpack_sequence, unpad_sequence

test_cases = [
    [torch.randn(abs(n - 3)) for n in range(6)],
    # [torch.randn(abs(n - 3), 2) for n in range(6)],
    # [torch.randn(abs(n - 3), 2, 3) for n in range(6)],
]


@mark.parametrize("tensors", test_cases)
def test_unpack_sequence(tensors: list[Tensor]):
    r"""Test if `unpack_sequence` is inverse to `torch.nn.utils.rnn.pack_sequence`."""
    tensors = [t for t in tensors if len(t) > 0]

    packed = pack_sequence(tensors, enforce_sorted=False)
    unpacked = unpack_sequence(packed)

    assert len(unpacked) == len(tensors)

    for x, y in zip(tensors, unpacked):
        assert torch.all(x == y)


@mark.parametrize("tensors", test_cases)
def test_unpad_lengths(tensors: list[Tensor]) -> None:
    r"""Test if `test_unpad` is inverse to `torch.nn.utils.rnn.pad_sequence`."""
    padding_value = float("nan")
    lengths = [len(t) for t in tensors]

    for i, t in enumerate(tensors):
        if len(t) > 0:
            tensors[i][0] = padding_value

    padded_seq = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    unpadded = unpad_sequence(padded_seq, batch_first=True, lengths=lengths)

    assert len(unpadded) == len(tensors)

    for x, y in zip(tensors, unpadded):
        mask_x = torch.isnan(x)
        mask_y = torch.isnan(y)
        assert torch.all(mask_x == mask_y)
        assert torch.all(x[~mask_x] == y[~mask_y])


@mark.parametrize("tensors", test_cases)
def test_unpad_sequence_nan(tensors: list[Tensor]) -> None:
    r"""Test if `test_unpad` is inverse to `torch.nn.utils.rnn.pad_sequence`."""
    padding_value = float("nan")

    for i, t in enumerate(tensors):
        if len(t) > 0:
            tensors[i][0] = padding_value

    padded_seq = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    unpadded = unpad_sequence(padded_seq, batch_first=True, padding_value=padding_value)

    assert len(unpadded) == len(tensors)

    for x, y in zip(tensors, unpadded):
        mask_x = torch.isnan(x)
        mask_y = torch.isnan(y)
        assert torch.all(mask_x == mask_y)
        assert torch.all(x[~mask_x] == y[~mask_y])


@mark.parametrize("tensors", test_cases)
def test_unpad_sequence_float(tensors: list[Tensor]) -> None:
    r"""Test if `test_unpad` is inverse to `torch.nn.utils.rnn.pad_sequence`."""
    padding_value = 0.0

    for i, t in enumerate(tensors):
        if len(t) > 0:
            tensors[i][0] = padding_value

    padded_seq = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    unpadded = unpad_sequence(padded_seq, batch_first=True, padding_value=padding_value)

    assert len(unpadded) == len(tensors)

    for x, y in zip(tensors, unpadded):
        mask_x = x == padding_value
        mask_y = y == padding_value
        assert torch.all(mask_x == mask_y)
        assert torch.all(x[~mask_x] == y[~mask_y])


if __name__ == "__main__":
    # main program
    test_unpack_sequence()
    test_unpad_sequence_nan()
    test_unpad_sequence_float()