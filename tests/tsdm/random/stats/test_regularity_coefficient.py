r"""Test for checking how regular time series is."""

from collections.abc import Callable

import numpy as np
import pytest

DATA_EXAMPLES = {
    "almost-regular": np.array([0, 1, 2, 3, 5, 6, 7, 8, 9]),
    # "long-irregular" : ...,
    # "long-regular" : ...,
    "short-irregular": np.array([0, 12, 20]),
    "short-regular": np.array([0, 10, 20]),
}


@pytest.mark.parametrize("gamma", [...])
@pytest.mark.parametrize("example", DATA_EXAMPLES)
def test_shift_invariance(example: str, gamma: Callable) -> None:
    data = DATA_EXAMPLES[example]
    assert gamma(data) == gamma(data + 1)


@pytest.mark.parametrize("gamma", [...])
@pytest.mark.parametrize("example", DATA_EXAMPLES)
def test_scale_invariance(example: str, gamma: Callable) -> None:
    data = DATA_EXAMPLES[example]
    assert gamma(data) == gamma(data * 2)
