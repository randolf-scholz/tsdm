from math import prod

import pytest
import torch

from tsdm.metrics.sequential import SequentialMSE

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
NUM_STEPS = [1, 32]


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("channel_shape", CHANNEL_SHAPES, ids=lambda cs: f"{cs=}")
@pytest.mark.parametrize("num_steps", NUM_STEPS)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=lambda bs: f"{bs=}")
@pytest.mark.parametrize(("atol", "rtol"), [(0.01, 0.01)])
def test_time_loss_normalization(
    batch_shape: tuple[int, ...],
    num_steps: int,
    channel_shape: tuple[int, ...],
    atol: float,
    rtol: float,
) -> None:
    r"""Test whether the time-series losses are normalized.

    Note:
        The expectation is that, if $x,x̂∼N(0,1)$, then $r=x-x̂ ∼ N(0,2)$.
        Thus, $‖r‖^2  = ‖√2⋅p‖^2 = 2‖p‖^2 = 2⋅χ^2(K) = χ^2(2K)$.

        The chi-squared distribution strongly concentrates around its mean,
        so we should expect to see a value close to $2K$.
    """
    shape = (*batch_shape, num_steps, *channel_shape)
    targets = torch.randn(*shape)
    predictions = torch.randn(*shape)

    K = len(channel_shape)
    time_axis = -K - 1
    channel_axes = tuple(range(-K, 0))
    print(shape, time_axis, channel_axes)
    loss_func = SequentialMSE(
        time_dim=time_axis,
        channel_dim=channel_axes,
        normalize_time=True,
        normalize=False,
    )

    result = loss_func(predictions=predictions, targets=targets)

    # skip for edge case test
    if prod(batch_shape) <= 1 or num_steps <= 1:
        return

    expected = 2 * prod(channel_shape)
    assert abs(result - expected) < rtol * abs(expected) + atol, (
        f"tolerance exceeded! {shape=}, {result=}, {expected=}"
    )
