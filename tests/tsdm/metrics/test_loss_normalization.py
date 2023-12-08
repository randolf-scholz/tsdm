r"""Test loss function normalization."""

from math import pi, prod, sqrt

import pytest
import torch

from tsdm.metrics import MAE, MSE, RMSE, BaseMetric, TimeSeriesMSE

BATCH_SHAPES = [
    (),
    (1,),
    (128,),
    (8, 16),
]
CHANNEL_SHAPES = [
    (64,),
    (5, 16),
]
TIME_SHAPES = [(1,), (32,)]
LOSSES = [MSE, RMSE, MAE]


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("channel_shape", CHANNEL_SHAPES, ids=lambda cs: f"{cs=}")
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=lambda bs: f"{bs=}")
def test_loss_normalization(
    loss: type[BaseMetric],
    batch_shape: tuple[int, ...],
    channel_shape: tuple[int, ...],
    atol: float = 0.01,
    rtol: float = 0.01,
) -> None:
    r"""Test whether the modular losses are normalized."""
    loss_func = loss(normalize=True)
    shape = batch_shape + channel_shape
    targets = torch.randn(*shape)
    predictions = torch.randn(*shape)
    result = loss_func(targets, predictions)

    if prod(batch_shape) <= 1:
        return

    expected = {
        "MAE": 2 / sqrt(pi),
        "MSE": 2,
        "RMSE": sqrt(2),
    }[loss.__name__]
    assert (
        abs(result - expected) < rtol * abs(expected) + atol
    ), f"tolerance exceeded! {shape=}, {result=}, {expected=}"


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("channel_shape", CHANNEL_SHAPES, ids=lambda cs: f"{cs=}")
@pytest.mark.parametrize("time_shape", TIME_SHAPES, ids=lambda ts: f"{ts=}")
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=lambda bs: f"{bs=}")
def test_time_loss_normalization(
    batch_shape: tuple[int, ...],
    time_shape: tuple[int, ...],
    channel_shape: tuple[int, ...],
    atol: float = 0.01,
    rtol: float = 0.01,
) -> None:
    r"""Test whether the time-series losses are normalized.

    Note:
        The expectation is that, if $x,x̂∼N(0,1)$, then $r=x-x̂ ∼ N(0,2)$.
        Thus, $‖r‖^2  = ‖√2⋅p‖^2 = 2‖p‖^2 = 2⋅χ^2(K) = χ^2(2K)$.

        The chi-squared distribution strongly concentrates around its mean,
        so we should expect to see a value close to $2K$.
    """
    shape = batch_shape + time_shape + channel_shape
    targets = torch.randn(*shape)
    predictions = torch.randn(*shape)

    T, K = len(time_shape), len(channel_shape)
    channel_axes = tuple(range(-K, 0))
    time_axes = tuple(range(-T - K, -K))
    print(shape, time_axes, channel_axes)
    loss_func = TimeSeriesMSE(
        time_axis=time_axes,
        channel_axis=channel_axes,
        normalize_time=True,
        normalize_channels=False,
    )

    result = loss_func(targets, predictions)

    # skip for edge case test
    if prod(batch_shape) <= 1 or prod(time_shape) <= 1:
        return

    expected = 2 * prod(channel_shape)
    assert (
        abs(result - expected) < rtol * abs(expected) + atol
    ), f"tolerance exceeded! {shape=}, {result=}, {expected=}"
