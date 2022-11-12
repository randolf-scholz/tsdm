#!/usr/bin/env python
r"""Test RNN utils."""

import logging

import pytest
import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from tsdm.utils.data import unpack_sequence, unpad_sequence

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


TESTCASES = [
    # [torch.randn(abs(n - 3)) for n in range(6)], #FIXME: https://github.com/pytorch/pytorch/issues/80605
    [torch.randn(abs(n - 3), 2) for n in range(6)],
    [torch.randn(abs(n - 3), 2, 3) for n in range(6)],
]


@pytest.mark.parametrize("tensors", TESTCASES)
def test_unpack_sequence(tensors: list[Tensor]) -> None:
    r"""Test if `unpack_sequence` is inverse to `torch.nn.utils.rnn.pack_sequence`."""
    LOGGER = __logger__.getChild(unpack_sequence.__name__)
    LOGGER.info("Testing.")

    tensors = [t for t in tensors if len(t) > 0]

    packed = pack_sequence(tensors, enforce_sorted=False)
    unpacked = unpack_sequence(packed)

    assert len(unpacked) == len(tensors)

    for x, y in zip(tensors, unpacked):
        assert torch.all(x == y)


@pytest.mark.parametrize("tensors", TESTCASES)
def test_unpad_lengths(tensors: list[Tensor]) -> None:
    r"""Test if `test_unpad` is inverse to `torch.nn.utils.rnn.pad_sequence`."""
    LOGGER = __logger__.getChild(unpad_sequence.__name__)
    LOGGER.info("Testing.")

    padding_value = float("nan")
    lengths = torch.tensor([len(t) for t in tensors], dtype=torch.int32)

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


@pytest.mark.parametrize("tensors", TESTCASES)
def test_unpad_sequence_nan(tensors: list[Tensor]) -> None:
    r"""Test if `test_unpad` is inverse to `torch.nn.utils.rnn.pad_sequence`."""
    LOGGER = __logger__.getChild(unpad_sequence.__name__)
    LOGGER.info("Testing with NaN padding.")

    padding_value = float("nan")

    for i, t in enumerate(tensors):
        if len(t) > 0:
            tensors[i][0] = padding_value
    print(tensors)
    # if len(tensors)
    padded_seq = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    unpadded = unpad_sequence(padded_seq, batch_first=True, padding_value=padding_value)

    assert len(unpadded) == len(tensors)

    for x, y in zip(tensors, unpadded):
        mask_x = torch.isnan(x)
        mask_y = torch.isnan(y)
        assert torch.all(mask_x == mask_y)
        assert torch.all(x[~mask_x] == y[~mask_y])


@pytest.mark.parametrize("tensors", TESTCASES)
def test_unpad_sequence_float(tensors: list[Tensor]) -> None:
    r"""Test if `test_unpad` is inverse to `torch.nn.utils.rnn.pad_sequence`."""
    LOGGER = __logger__.getChild(unpad_sequence.__name__)
    LOGGER.info("Testing with float padding.")

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


def _main() -> None:
    for tensors in TESTCASES:
        test_unpack_sequence(tensors)
        test_unpad_lengths(tensors)
        test_unpad_sequence_nan(tensors)
        test_unpad_sequence_float(tensors)


if __name__ == "__main__":
    _main()
