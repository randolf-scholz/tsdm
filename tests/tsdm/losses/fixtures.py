from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import Generator, Tensor


@dataclass(frozen=True, slots=True)
class TestCase:
    targets: Tensor
    predictions: Tensor
    mask: Tensor | None


class TestSequenceLoss:
    r"""Shared tests for functional sequence losses."""

    loss: Callable[..., Tensor]

    SEQUENCE_LENGTH = 4

    def make_test_case(
        self,
        *,
        batch_shape: tuple[int, ...],
        sequence_length: int,
        channel_shape: tuple[int, ...],
        sparse_targets: bool,
        rng: Generator | int | None,
        device: torch.device | None = None,
    ) -> TestCase:
        r"""Generate a pair of targets and predictions."""
        shape = (*batch_shape, sequence_length, *channel_shape)
        generator = (
            rng
            if isinstance(rng, Generator)
            else (
                torch.Generator() if device is None else torch.Generator(device=device)
            ).manual_seed(0 if rng is None else rng)
        )
        targets = torch.randn(shape, generator=generator, device=device)
        predictions = torch.randn(shape, generator=generator, device=device)

        if not sparse_targets:
            return TestCase(targets=targets, predictions=predictions, mask=None)

        mask = torch.rand(shape, generator=generator, device=device) > 0.5
        mask.select(dim=len(batch_shape), index=0).fill_(value=True)
        targets = targets.masked_fill(~mask, torch.nan)
        return TestCase(targets=targets, predictions=predictions, mask=mask)

    @pytest.mark.parametrize("batch_shape", [(), (6,), (1, 2, 3)])
    def test_univariate_series(self, batch_shape: tuple[int, ...]) -> None:
        r"""Univariate inputs use ``channel_dim=()`` without reducing batch axes."""
        case = self.make_test_case(
            batch_shape=batch_shape,
            sequence_length=self.SEQUENCE_LENGTH,
            channel_shape=(),
            sparse_targets=True,
            rng=0,
        )

        result = self.loss(
            predictions=case.predictions,
            targets=case.targets,
            mask=case.mask,
            time_dim=-1,
            channel_dim=(),
        )

        assert result.ndim == 0
        assert torch.isfinite(result)

    @pytest.mark.parametrize("batch_shape", [(), (6,), (1, 2, 3)])
    @pytest.mark.parametrize("channel_shape", [(), (6,), (1, 2, 3)])
    def test_backprop_predictions(
        self,
        batch_shape: tuple[int, ...],
        channel_shape: tuple[int, ...],
    ) -> None:
        r"""Masked target NaNs do not prevent finite prediction gradients."""
        case = self.make_test_case(
            batch_shape=batch_shape,
            sequence_length=self.SEQUENCE_LENGTH,
            channel_shape=channel_shape,
            sparse_targets=True,
            rng=0,
        )
        predictions = case.predictions.requires_grad_()
        channel_dim = tuple(range(-len(channel_shape), 0))

        result = self.loss(
            predictions=predictions,
            targets=case.targets,
            mask=case.mask,
            time_dim=-len(channel_shape) - 1,
            channel_dim=channel_dim,
        )
        result.backward()

        assert torch.isfinite(result)
        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()
        assert case.mask is not None
        assert torch.equal(
            predictions.grad[~case.mask],
            torch.zeros_like(predictions.grad[~case.mask]),
        )
