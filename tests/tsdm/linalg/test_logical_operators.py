r"""Tests for logical operators."""

import pytest
import torch
from torch import Tensor

from tsdm.linalg import cumulative_and, cumulative_or

DEVICES = ["cpu"] + ["cuda"] * torch.cuda.is_available()


@pytest.mark.parametrize(
    ("x", "dim", "expected"),
    [
        (
            torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=torch.bool),
            0,
            torch.tensor([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.bool),
        ),
        (
            torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=torch.bool),
            1,
            torch.tensor([[1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool),
        ),
        (
            torch.tensor([[0, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool),
            0,
            torch.tensor([[0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool),
        ),
        (
            torch.tensor([[0, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool),
            1,
            torch.tensor([[0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool),
        ),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_cumulative_or(x: Tensor, dim: int, device: str, expected: Tensor) -> None:
    dev = torch.device(device)
    x = x.to(dev)
    expected = expected.to(dev)
    result = cumulative_or(x, dim)
    assert torch.all(result == expected)


@pytest.mark.parametrize(
    ("x", "dim", "expected"),
    [
        (
            torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=torch.bool),
            0,
            torch.tensor([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool),
        ),
        (
            torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=torch.bool),
            1,
            torch.tensor([[1, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.bool),
        ),
        (
            torch.tensor([[0, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool),
            0,
            torch.tensor([[0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.bool),
        ),
        (
            torch.tensor([[0, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool),
            1,
            torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=torch.bool),
        ),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_cumulative_and(x: Tensor, dim: int, device: str, expected: Tensor) -> None:
    dev = torch.device(device)
    x = x.to(dev)
    expected = expected.to(dev)
    result = cumulative_and(x, dim)
    assert torch.all(result == expected)
