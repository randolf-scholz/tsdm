r"""Test for checking how regular time series is."""

from collections.abc import Callable

import numpy as np
import pytest

from tsdm.random.stats.regularity_tests import (
    coefficient_of_variation,
    geometric_std,
    irregularity_coefficient,
)

DATA_EXAMPLES: dict[str, np.ndarray] = {
    "almost-regular": np.array([0, 1, 2, 3, 5, 6, 7, 8, 9], dtype=int),
    # "long-irregular" : ...,
    # "long-regular" : ...,
    "short-irregular": np.array([0, 12, 20], dtype=int),
    "short-regular": np.array([0, 10, 20], dtype=int),
}


COEFFICIENTS: dict[str, Callable[[np.ndarray], float]] = {
    "coefficient_of_variation": coefficient_of_variation,
    "geometric_std": geometric_std,
    "irregularity_coefficient": irregularity_coefficient,
}


@pytest.mark.parametrize("coeff", COEFFICIENTS)
@pytest.mark.parametrize("example", DATA_EXAMPLES)
def test_shift_invariance(*, example: str, coeff: str) -> None:
    gamma = COEFFICIENTS[coeff]
    data = DATA_EXAMPLES[example]
    assert gamma(data) == gamma(data + 1)


@pytest.mark.parametrize("coeff", COEFFICIENTS)
@pytest.mark.parametrize("example", DATA_EXAMPLES)
def test_scale_invariance(*, example: str, coeff: str) -> None:
    gamma = COEFFICIENTS[coeff]
    data = DATA_EXAMPLES[example]
    assert gamma(data) == gamma(data * 2)
