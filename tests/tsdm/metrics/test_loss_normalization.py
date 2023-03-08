#!/usr/bin/env python
r"""Test loss function normalization."""

from itertools import product
from math import pi, prod, sqrt

import pytest
import torch

from tsdm.metrics import MAE, MSE, RMSE, BaseLoss, TimeSeriesMSE

BATCH_SHAPES = [
    (),
    (1,),
    (32,),
    # (64,),
    (4, 16),
]
CHANNEL_SHAPES = [
    (64,),
    # (128,),
    (8, 16),
]
TIME_SHAPES = [(1,), (32,)]
LOSSES = [MSE, RMSE, MAE]


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("channel_shape", CHANNEL_SHAPES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_loss_normalization(
    loss: type[BaseLoss],
    batch_shape: tuple[int, ...],
    channel_shape: tuple[int, ...],
    atol: float = 0.01,
    rtol: float = 0.01,
) -> None:
    r"""Test whether the modular losses are normalized."""
    l = loss(normalize=True)
    shape = batch_shape + channel_shape
    targets = torch.randn(*shape)
    predictions = torch.randn(*shape)
    result = l(targets, predictions)

    if prod(batch_shape) < 10:
        return

    expected = 2 / sqrt(pi) if loss == MAE else 1
    assert (
        abs(result - expected) < rtol * abs(expected) + atol
    ), f"tolerance exceeded! {shape=}, {result=}, {expected=}"


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("time_shape", TIME_SHAPES)
@pytest.mark.parametrize("channel_shape", CHANNEL_SHAPES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_time_loss_normalization(
    batch_shape: tuple[int, ...],
    time_shape: tuple[int, ...],
    channel_shape: tuple[int, ...],
    atol: float = 0.01,
    rtol: float = 0.01,
) -> None:
    r"""Test whether the time-series losses are normalized."""
    axes = tuple(range(-len(channel_shape), 0))

    l = TimeSeriesMSE(axes)
    shape = batch_shape + time_shape + channel_shape
    targets = torch.randn(*shape)
    predictions = torch.randn(*shape)
    result = l(targets, predictions)

    if prod(batch_shape) < 10:
        return

    expected = prod(channel_shape)
    assert (
        abs(result - expected) < rtol * abs(expected) + atol
    ), f"tolerance exceeded! {shape=}, {result=}, {expected=}"


def _main() -> None:
    r"""Main function."""
    for loss, batch_shape, channel_shape in product(
        LOSSES, BATCH_SHAPES, CHANNEL_SHAPES
    ):
        test_loss_normalization(loss, batch_shape, channel_shape)  # type: ignore[type-abstract]


if __name__ == "__main__":
    _main()
