r"""Test loss function normalization."""

from collections.abc import Callable
from math import pi, prod, sqrt

import pytest
import torch

from tsdm.losses import ND, BaseLoss, MAE_Loss, MSE_Loss
from tsdm.losses.batch import RMSE_Loss
from tsdm.losses.sequential import nd

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
LOSSES = [MSE_Loss, RMSE_Loss, MAE_Loss]


@pytest.mark.parametrize("loss_func", [nd, ND()])
def test_metric_argument_order(loss_func: Callable[..., torch.Tensor]) -> None:
    r"""Test that metrics accept predictions before targets by position or keyword."""
    targets = torch.ones(2, 2)
    predictions = 2 * torch.ones(2, 2)
    expected = torch.tensor(1.0)

    assert torch.equal(
        loss_func(predictions=predictions, targets=targets),
        expected,
    )


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("channel_shape", CHANNEL_SHAPES, ids=lambda cs: f"{cs=}")
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=lambda bs: f"{bs=}")
@pytest.mark.parametrize(("atol", "rtol"), [(0.01, 0.01)])
def test_loss_normalization(
    loss: type[BaseLoss],
    batch_shape: tuple[int, ...],
    channel_shape: tuple[int, ...],
    atol: float,
    rtol: float,
) -> None:
    r"""Test whether the modular losses are normalized."""
    loss_func = loss(scaled=True)
    shape = batch_shape + channel_shape
    targets = torch.randn(*shape)
    predictions = torch.randn(*shape)
    result = loss_func(predictions=predictions, targets=targets)

    if prod(batch_shape) <= 1:
        return

    expected = {
        "MAE_Loss": 2 / sqrt(pi),
        "MSE_Loss": 2,
        "RMSE_Loss": sqrt(2),
    }[loss.__name__]
    assert abs(result - expected) < rtol * abs(expected) + atol, (
        f"tolerance exceeded! {shape=}, {result=}, {expected=}"
    )
