r"""Test the generic rounding method."""

import math

import pytest
import torch

from tsdm.backend import generic

EXAMPLES = {
    "positive" : torch.tensor([
        0.00, 0.25, 0.50, 0.75,
        1.00, 1.25, 1.50, 1.75,
        2.00, 2.25, 2.50, 2.75,
        # 94.0000, 95.0000, 100.0000, 100.1000,
    ]),
    "negative": torch.tensor([
        # -100.1000, -100.0000, -95.0000, -94.0000,
        -2.00, -1.75, -1.50, -1.25,
        -1.00, -0.75, -0.50, -0.25,
    ]),
    "zero": torch.tensor([0.0, -0.0]),
    "special": torch.tensor([math.inf, math.nan, -math.inf]),
}  # fmt: skip


@pytest.mark.parametrize("decimals", [-1, 0, 1, 2])
@pytest.mark.parametrize("example", EXAMPLES)
def test_generic_round(example: str, decimals: int) -> None:
    data = EXAMPLES[example]
    expected = data.round(decimals=decimals)
    result = generic.round_impl(data, decimals=decimals)
    # assert all(result == expected), f"Expected {expected}, but got {result}."
    # FIXME: https://github.com/pytorch/pytorch/issues/137337
    torch.testing.assert_close(result, expected, equal_nan=True, atol=0.0, rtol=0.0)
