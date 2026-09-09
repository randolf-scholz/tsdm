from math import prod

import pytest
import torch

from tsdm.metrics.samplewise import Reduction
from tsdm.metrics.sequential import Normalization, SequentialMSE, lp_loss, lp_norm

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

        result = lp_norm(
            x,
            p=p,
            time_dim=-2,
            channel_dim=-1,
            normalization=None,
        )
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

        result = lp_norm(
            x,
            p=p,
            time_dim=time_dim,
            channel_dim=channel_dim,
            normalization=None,
        )
        expected = torch.linalg.vector_norm(x, ord=p, dim=(time_dim, *channel_dim))

        assert result.shape == batch_shape
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("p", P_VALUES)
    @pytest.mark.parametrize(
        ("shape", "time_dim", "channel_dim", "weight_shape"),
        [
            ((7, 2, 3, 4), 0, (-2, -1), (1, 1, 3, 4)),
            ((2, 3, 4, 7), -1, (-3, -2), (1, 3, 4, 1)),
        ],
        ids=["time-batch-channel", "batch-channel-time"],
    )
    def test_channel_weights_align_with_noncanonical_dimension_orders(
        self,
        p: float,
        shape: tuple[int, ...],
        time_dim: int,
        channel_dim: tuple[int, ...],
        weight_shape: tuple[int, ...],
    ) -> None:
        r"""Channel weights align with their declared dimensions, not just trailing ones."""
        x = torch.linspace(0.1, 2.0, prod(shape), dtype=torch.float64).reshape(shape)
        channel_weight = torch.arange(1.0, 13.0, dtype=x.dtype).reshape(3, 4)

        result = lp_norm(
            x,
            p=p,
            time_dim=time_dim,
            channel_dim=channel_dim,
            channel_weight=channel_weight,
            normalization=None,
        )
        expected = (
            x.abs()
            .pow(p)
            .mul(channel_weight.reshape(weight_shape))
            .sum(dim=(time_dim, *channel_dim))
            .pow(1 / p)
        )

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        ("channel_dim", "time_dim"),
        [((1, 0), -1), ((-1, -2), 0)],
    )
    def test_rejects_nonincreasing_channel_dimensions(
        self,
        channel_dim: tuple[int, ...],
        time_dim: int,
    ) -> None:
        r"""Channel dimensions must follow their physical tensor-axis order."""
        x = torch.ones(2, 3, 4)

        with pytest.raises(ValueError, match="strictly increasing"):
            lp_norm(x, time_dim=time_dim, channel_dim=channel_dim)

    def test_rejects_scalar_time_weight(self) -> None:
        r"""A time weight must include the time dimension."""
        x = torch.ones(2, 3, 4)

        with pytest.raises(ValueError, match=r"1 <= w_t.ndim"):
            lp_norm(
                x,
                time_dim=-2,
                channel_dim=-1,
                time_weight=torch.tensor(1.0),
            )

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
            normalization=None,
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
    @pytest.mark.parametrize("normalization", Normalization)
    def test_normalization(self, p: float, normalization: Normalization) -> None:
        r"""Each normalization scheme uses its corresponding masked observation count."""
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
            normalization=normalization,
        )
        match normalization:
            case Normalization.SEQUENCE_LENGTH:
                counts = (
                    mask.any(dim=-1, keepdim=True)
                    .sum(dim=-2, keepdim=True)
                    .clamp_min(1)
                )
            case Normalization.CHANNEL_PREVALENCE:
                counts = mask.sum(dim=-2, keepdim=True).clamp_min(1)
            case Normalization.TIMEPOINT_COVERAGE:
                counts = mask.any(dim=-1, keepdim=True).sum(
                    dim=-2, keepdim=True
                ).clamp_min(1) * mask.sum(dim=-1, keepdim=True).clamp_min(1)
            case Normalization.OBSERVATION_COUNT:
                counts = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        expected = (
            (torch.where(mask, x.abs().pow(p), 0.0) / counts)
            .sum(dim=(-2, -1))
            .pow(1 / p)
        )

        torch.testing.assert_close(result, expected)

    def test_defaults_to_channel_prevalence_normalization(self) -> None:
        r"""Channel-prevalence normalization is the default."""
        x = torch.arange(1.0, 7.0).reshape(1, 2, 3)
        mask = torch.tensor([[[True, False, True], [True, True, False]]])

        result = lp_norm(x, mask=mask, time_dim=-2, channel_dim=-1)
        expected = lp_norm(
            x,
            mask=mask,
            time_dim=-2,
            channel_dim=-1,
            normalization=Normalization.CHANNEL_PREVALENCE,
        )

        torch.testing.assert_close(result, expected)

    def test_custom_normalization_is_not_implemented(self) -> None:
        r"""Custom normalization tensors are reserved for a future implementation."""
        x = torch.ones(2, 3, 4)

        with pytest.raises(NotImplementedError):
            lp_norm(
                x,
                time_dim=-2,
                channel_dim=-1,
                normalization=torch.ones(1),
            )

    @pytest.mark.parametrize("p", P_VALUES)
    @pytest.mark.parametrize("normalization", Normalization)
    def test_missing_mask_equals_all_observed_mask(
        self,
        p: float,
        *,
        normalization: Normalization,
    ) -> None:
        r"""An omitted mask treats every value as observed."""
        x = torch.linspace(0.1, 2.4, 24).reshape(2, 3, 4)

        result = lp_norm(
            x,
            p=p,
            time_dim=-2,
            channel_dim=-1,
            normalization=normalization,
        )
        expected = lp_norm(
            x,
            p=p,
            mask=torch.ones_like(x, dtype=torch.bool),
            time_dim=-2,
            channel_dim=-1,
            normalization=normalization,
        )

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("p", P_VALUES)
    @pytest.mark.parametrize("normalization", Normalization)
    def test_all_masked_values_return_zero(
        self,
        p: float,
        normalization: Normalization,
    ) -> None:
        r"""An all-zero mask produces zero norms, including for negative orders."""
        x = torch.linspace(0.1, 2.4, 24).reshape(2, 3, 4)
        mask = torch.zeros_like(x, dtype=torch.bool)

        result = lp_norm(
            x,
            p=p,
            mask=mask,
            time_dim=-2,
            channel_dim=-1,
            normalization=normalization,
        )

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


class TestLpLoss:
    r"""Tests for sequential Lp losses."""

    @pytest.mark.parametrize("reduction", list(Reduction))
    def test_relative_divides_residual_norm_by_target_norm(
        self, reduction: Reduction
    ) -> None:
        r"""Relative losses are normalized separately for each sequence."""
        predictions = torch.tensor(
            [[[2.0, 4.0], [6.0, 8.0]], [[3.0, 6.0], [9.0, 12.0]]]
        )
        targets = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]])

        result = lp_loss(
            predictions=predictions,
            targets=targets,
            p=2.0,
            time_dim=-2,
            channel_dim=-1,
            normalization=None,
            reduction=reduction,
            relative=True,
        )
        ratios = lp_norm(
            predictions - targets,
            p=2.0,
            time_dim=-2,
            channel_dim=-1,
            normalization=None,
        ) / lp_norm(
            targets,
            p=2.0,
            time_dim=-2,
            channel_dim=-1,
            normalization=None,
        )
        expected = {
            Reduction.NONE: ratios,
            Reduction.SUM: ratios.sum(),
            Reduction.MEAN: ratios.mean(),
        }[reduction]

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("reduction", list(Reduction))
    def test_matches_lp_norm_of_residuals(self, reduction: Reduction) -> None:
        r"""The loss wraps the corresponding norm and then reduces it."""
        predictions = torch.arange(24.0).reshape(2, 3, 4)
        targets = torch.arange(24.0, 0.0, -1.0).reshape(2, 3, 4)

        result = lp_loss(
            predictions=predictions,
            targets=targets,
            p=1.0,
            time_dim=-2,
            channel_dim=-1,
            normalization=None,
            reduction=reduction,
        )
        norms = lp_norm(
            predictions - targets,
            p=1.0,
            time_dim=-2,
            channel_dim=-1,
            normalization=None,
        )
        expected = {
            Reduction.NONE: norms,
            Reduction.SUM: norms.sum(),
            Reduction.MEAN: norms.mean(),
        }[reduction]

        torch.testing.assert_close(result, expected)

    def test_defaults_to_mean_reduction(self) -> None:
        r"""The default reduction averages the per-sequence norms."""
        predictions = torch.arange(24.0).reshape(2, 3, 4)
        targets = torch.arange(24.0, 0.0, -1.0).reshape(2, 3, 4)

        result = lp_loss(
            predictions=predictions,
            targets=targets,
            time_dim=-2,
            channel_dim=-1,
        )
        expected = lp_norm(
            predictions - targets,
            time_dim=-2,
            channel_dim=-1,
        ).mean()

        torch.testing.assert_close(result, expected)

    def test_masks_sparse_targets_without_breaking_prediction_gradients(self) -> None:
        r"""Sparse targets do not contaminate prediction gradients."""
        predictions = torch.tensor(
            [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]], requires_grad=True
        )
        targets = torch.tensor([[[1.0, torch.nan, 3.0], [torch.nan, 5.0, 6.0]]])
        mask = ~targets.isnan()

        loss = lp_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=-2,
            channel_dim=-1,
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()
        assert torch.all(predictions.grad[mask] != 0)
        assert torch.equal(
            predictions.grad[~mask], torch.zeros_like(predictions.grad[~mask])
        )


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
