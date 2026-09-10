r"""Tests for sample-wise Lp norms."""

from collections.abc import Callable
from math import prod

import pytest
import torch
from torch.nn import functional

from tsdm.losses.batch import rmse_loss
from tsdm.losses.samplewise import (
    Reduction,
    lp_loss,
    lp_norm,
    mae_loss,
    mse_loss,
)


class TestLpNorm:
    @pytest.mark.parametrize("p", [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("dim", [-1, (1, 2), None])
    def test_matches_torch_vector_norm(
        self,
        p: float,
        dim: int | tuple[int, ...] | None,
    ) -> None:
        r"""The ordinary unscaled norm agrees with PyTorch for finite non-zero inputs."""
        x = torch.linspace(0.1, 2.4, 24, dtype=torch.float64).reshape(2, 3, 4)

        result = lp_norm(x, p=p, dim=dim)
        expected = torch.linalg.vector_norm(x, ord=p, dim=dim)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("weight_shape", [(4,), (3, 4)])
    def test_weight_matches_reduction_axes(
        self,
        weight_shape: tuple[int, ...],
    ) -> None:
        r"""Weights span precisely the explicitly reduced axes."""
        x = torch.linspace(0.1, 2.4, 24, dtype=torch.float64).reshape(2, 3, 4)
        weight = torch.ones(weight_shape, dtype=x.dtype)
        dim = tuple(range(-weight.ndim, 0))

        result = lp_norm(x, dim=dim, weight=weight)
        expected = torch.linalg.vector_norm(x, dim=dim)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [-1, (1, 2), None])
    def test_scaled_divides_by_reduced_size(
        self,
        dim: int | tuple[int, ...] | None,
    ) -> None:
        r"""Scaling uses the number of elements along the reduced axes."""
        x = torch.linspace(0.1, 2.4, 24, dtype=torch.float64).reshape(2, 3, 4)
        dims = (
            tuple(range(x.ndim))
            if dim is None
            else (dim,)
            if isinstance(dim, int)
            else dim
        )

        result = lp_norm(x, dim=dim, scaled=True)
        expected = (
            torch.linalg.vector_norm(x, dim=dim) / prod(x.shape[d] for d in dims) ** 0.5
        )

        torch.testing.assert_close(result, expected)

    def test_masked_negative_excludes_masked_values(self) -> None:
        r"""Masked values do not contribute, including when p is negative."""
        x = torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
        mask = torch.tensor([[True, False, True], [False, True, True]])

        result = lp_norm(x, p=-1.0, dim=-1, mask=mask)
        expected = torch.tensor([1 / (1 + 1 / 4), 1 / (1 / 4 + 1 / 8)])

        torch.testing.assert_close(result, expected)

    def test_scaled_weighted_excludes_zero_weight_entries(self) -> None:
        r"""Zero-weight entries do not contribute to the normalization count."""
        x = torch.tensor([3.0, 4.0, 5.0])
        mask = torch.tensor([True, True, False])
        weight = torch.tensor([1.0, 0.0, 1.0])

        result = lp_norm(x, dim=-1, mask=mask, weight=weight, scaled=True)

        torch.testing.assert_close(result, torch.tensor(3.0))

    def test_masks_nan_values_without_breaking_gradients(self) -> None:
        r"""Valid values retain finite, non-zero gradients in NaN-contaminated inputs."""
        values = torch.tensor(
            [[1.0, torch.nan, 2.0], [torch.nan, 3.0, 4.0]],
            requires_grad=True,
        )
        mask = ~values.isnan()

        loss = lp_norm(values, dim=-1, mask=mask).sum()
        loss.backward()

        assert torch.isfinite(loss)
        assert values.grad is not None
        assert torch.isfinite(values.grad[mask]).all()
        assert torch.all(values.grad[mask] != 0)


class TestLpLoss:
    @pytest.mark.parametrize("reduction", list(Reduction))
    def test_relative_divides_residual_norm_by_target_norm(
        self, reduction: Reduction
    ) -> None:
        r"""Relative losses are normalized separately for each sample."""
        predictions = torch.tensor([[2.0, 4.0], [6.0, 12.0]])
        targets = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

        result = lp_loss(
            predictions=predictions,
            targets=targets,
            p=2.0,
            reduction=reduction,
            relative=True,
        )
        ratios = torch.linalg.vector_norm(
            predictions - targets, dim=-1
        ) / torch.linalg.vector_norm(targets, dim=-1)
        expected = {
            Reduction.NONE: ratios,
            Reduction.SUM: ratios.sum(),
            Reduction.MEAN: ratios.mean(),
        }[reduction]

        torch.testing.assert_close(result, expected)

    def test_masks_nan_targets_without_breaking_prediction_gradients(
        self,
    ) -> None:
        r"""All predictions retain finite gradients with masked NaN targets."""
        predictions = torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            requires_grad=True,
        )
        targets = torch.tensor([[1.0, torch.nan, 3.0], [torch.nan, 5.0, 6.0]])
        mask = ~targets.isnan()

        loss = lp_loss(predictions=predictions, targets=targets, mask=mask)
        loss.backward()

        assert torch.isfinite(loss)
        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape
        assert torch.isfinite(predictions.grad).all()
        assert torch.all(predictions.grad[mask] != 0)
        assert torch.equal(
            predictions.grad[~mask], torch.zeros_like(predictions.grad[~mask])
        )

    def test_separates_sample_and_channel_weights(self) -> None:
        r"""Sample weights apply after channel-weighted norm reduction."""
        predictions = torch.zeros(2, 2)
        targets = torch.ones(2, 2)

        result = lp_loss(
            predictions=predictions,
            targets=targets,
            p=1.0,
            channel_weight=torch.tensor([1.0, 2.0]),
            weight=torch.tensor([1.0, 3.0]),
            reduction=Reduction.NONE,
        )

        torch.testing.assert_close(result, torch.tensor([3.0, 9.0]))


@pytest.mark.parametrize("reduction", list(Reduction))
@pytest.mark.parametrize(
    "weight",
    [None, torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])],
)
@pytest.mark.parametrize(
    ("loss", "torch_loss"),
    [(mae_loss, functional.l1_loss), (mse_loss, functional.mse_loss)],
)
def test_lp_losses_match_torch_functional(
    loss: Callable[..., torch.Tensor],
    torch_loss: Callable[..., torch.Tensor],
    reduction: Reduction,
    weight: torch.Tensor | None,
) -> None:
    r"""MAE and MSE agree with PyTorch when no inner axes are reduced."""
    predictions = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    targets = torch.tensor([[1.0, 2.0, 1.0], [5.0, 2.0, 8.0]])

    result = loss(
        predictions=predictions,
        targets=targets,
        dim=(),
        weight=weight,
        reduction=reduction,
    )
    expected = torch_loss(
        predictions, targets, reduction=reduction.value, weight=weight
    )

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("reduction", list(Reduction))
@pytest.mark.parametrize(
    "weight",
    [None, torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])],
)
def test_rmse_loss_matches_square_root_of_torch_mse_loss(
    reduction: Reduction,
    weight: torch.Tensor | None,
) -> None:
    r"""RMSE is the square root of the corresponding weighted MSE."""
    predictions = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    targets = torch.tensor([[1.0, 2.0, 1.0], [5.0, 2.0, 8.0]])

    result = rmse_loss(
        predictions=predictions,
        targets=targets,
        dim=(),
        weight=weight,
        reduction=reduction,
    )
    expected = functional.mse_loss(
        predictions,
        targets,
        reduction=reduction.value,
        weight=weight,  # type: ignore
    ).sqrt()

    torch.testing.assert_close(result, expected)
