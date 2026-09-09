from math import prod

import pytest
import torch

from tsdm.metrics.sequential import SequentialMSE, lp_norm

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

P_VALUES = [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0]
LP_NORM_BATCH_SHAPES = [(), (6,), (1, 2, 3)]
LP_NORM_CHANNEL_SHAPES = [(), (4,), (2, 2)]
LP_NORM_SEQUENCE_LENGTH = 7


class TestLpNorm:
    r"""Tests for sequential Lp norms."""

    @pytest.mark.parametrize("p", P_VALUES)
    def test_matches_torch_vector_norm(self, p: float) -> None:
        r"""The unscaled sequential norm agrees with PyTorch."""
        x = torch.linspace(0.1, 2.4, 24, dtype=torch.float64).reshape(2, 3, 4)

        result = lp_norm(x, p=p, time_dim=-2, channel_dim=-1)
        expected = torch.linalg.vector_norm(x, ord=p, dim=(-2, -1))

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("p", P_VALUES)
    @pytest.mark.parametrize("batch_shape", LP_NORM_BATCH_SHAPES)
    @pytest.mark.parametrize("channel_shape", LP_NORM_CHANNEL_SHAPES)
    def test_accepts_arbitrary_batch_and_channel_shapes(
        self,
        p: float,
        batch_shape: tuple[int, ...],
        channel_shape: tuple[int, ...],
    ) -> None:
        r"""Arbitrary batch and channel shapes include unbatched univariate series."""
        shape = (*batch_shape, LP_NORM_SEQUENCE_LENGTH, *channel_shape)
        x = torch.linspace(0.1, 2.0, prod(shape), dtype=torch.float64).reshape(shape)
        time_dim = -len(channel_shape) - 1
        channel_dim = tuple(range(-len(channel_shape), 0))

        result = lp_norm(x, p=p, time_dim=time_dim, channel_dim=channel_dim)
        expected = torch.linalg.vector_norm(x, ord=p, dim=(time_dim, *channel_dim))

        assert result.shape == batch_shape
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("p", P_VALUES)
    def test_masks_values_and_preserves_gradients(self, p: float) -> None:
        r"""Masks exclude values while valid inputs and weights remain differentiable."""
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
        mask = torch.tensor([[[True, False, True], [False, True, True]]])
        time_weight = torch.tensor([[1.0, 2.0]], requires_grad=True)
        channel_weight = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        result = lp_norm(
            x,
            p=p,
            mask=mask,
            time_dim=-2,
            channel_dim=-1,
            time_weight=time_weight,
            channel_weight=channel_weight,
        )
        expected = (
            (
                torch.where(mask, x.abs().pow(p), 0.0)
                * time_weight[..., None]
                * channel_weight
            )
            .sum(dim=(-2, -1))
            .pow(1 / p)
        )
        result.sum().backward()

        torch.testing.assert_close(result, expected)
        assert x.grad is not None
        assert time_weight.grad is not None
        assert channel_weight.grad is not None
        assert torch.isfinite(x.grad[mask]).all()
        assert torch.isfinite(time_weight.grad).all()
        assert torch.isfinite(channel_weight.grad).all()
        assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))

    @pytest.mark.parametrize("p", P_VALUES)
    def test_prevalence_scaling(self, p: float) -> None:
        r"""Each channel is scaled by its number of observations, safely at zero."""
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        mask = torch.tensor(
            [[[True, True, False], [True, False, False], [False, False, False]]]
        )

        result = lp_norm(
            x,
            p=p,
            mask=mask,
            time_dim=-2,
            channel_dim=-1,
            scale_channels=True,
        )
        counts = mask.sum(dim=-2, keepdim=True).clamp_min(1)
        expected = (
            (torch.where(mask, x.abs().pow(p), 0.0) / counts)
            .sum(dim=(-2, -1))
            .pow(1 / p)
        )

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("p", P_VALUES)
    def test_all_masked_values_return_zero(self, p: float) -> None:
        r"""An all-zero mask produces zero norms, including for negative orders."""
        x = torch.linspace(0.1, 2.4, 24).reshape(2, 3, 4)
        mask = torch.zeros_like(x, dtype=torch.bool)

        result = lp_norm(x, p=p, mask=mask, time_dim=-2, channel_dim=-1)

        torch.testing.assert_close(result, torch.zeros_like(result))

    @pytest.mark.parametrize("p", P_VALUES)
    def test_masked_nan_values_do_not_contaminate_norm_or_gradients(
        self, p: float
    ) -> None:
        r"""Masked NaNs are excluded before the power operation and backpropagation."""
        x = torch.tensor([[[1.0, torch.nan], [2.0, torch.nan]]], requires_grad=True)
        mask = ~x.isnan()

        result = lp_norm(x, p=p, mask=mask, time_dim=-2, channel_dim=-1)
        result.sum().backward()

        assert torch.isfinite(result).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad[mask]).all()
        assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))


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
