r"""Test the generic rounding method."""

import math

import pytest
import torch

from tsdm.backend import generic

EXAMPLES = {
    "positive" : torch.tensor([
        0.2500, 0.5000, 0.7500, 1.0000,
        1.2500, 1.5000, 1.7500, 2.0000,
        94.0000, 95.0000, 100.0000, 100.1000,
    ]),
    "negative": torch.tensor([
        -100.1000, -100.0000, -95.0000, -94.0000,
        -2.0000, -1.7500, -1.5000, -1.2500,
        -1.0000, -0.7500, -0.5000, -0.2500,
    ]),
    "zero": torch.tensor([0.0, -0.0]),
    "special": torch.tensor([math.inf, math.nan, -math.inf]),
}  # fmt: skip


@pytest.mark.parametrize("decimals", [-1, 0, 1, 2])
@pytest.mark.parametrize("example", EXAMPLES)
def test_round(example: str, decimals: int) -> None:
    data = EXAMPLES[example]
    expected = data.round(decimals=decimals)
    result = generic.round(data, decimals=decimals)
    assert all(result == expected)
