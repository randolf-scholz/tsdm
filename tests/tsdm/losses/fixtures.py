from dataclasses import dataclass

import pytest
from torch import Generator, Tensor

from tsdm.losses import SequenceLoss


@dataclass(frozen=True, slots=True)
class TestCase:
    targets: Tensor
    predictions: Tensor
    mask: Tensor | None


class TestSequenceLoss:
    loss: SequenceLoss

    SEQUENCE_LENGTH = 4

    def make_test_case(
        self,
        *,
        batch_shape: tuple[int, ...],
        sequence_length: int,
        channel_shape: tuple[int, ...],
        sparse_targets: bool,
        rng: Generator | int | None,
    ) -> TestCase:
        r"""Generate a pair of targets and predictions."""
        pass

    @pytest.mark.parametrize("batch_shape", [(), (6,), (1, 2, 3)])
    def test_univariate_series(self):
        # ensures that the case of univariate series of shape Float[..., $N]
        # is properly handled. in this case channel_dim=(), but there are some
        # footguns, e.g. torch.sum(dim=()) reduces all dimensions rather than none.
        pass

    @pytest.mark.parametrize("batch_shape", [(), (6,), (1, 2, 3)])
    @pytest.mark.parametrize("channel_shape", [(), (6,), (1, 2, 3)])
    def test_backprop_predictions(self):
        # ensure that gradients w.r.t. predictions propagate, even when targets are
        # contaminated with NaNs.
        pass
