r"""Test tsdm.plotting plotting utilities."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal
from numpy.typing import ArrayLike
from scipy.stats import gamma

from tests.test_utils import get_results_dir
from tsdm.plotting import plot_zero_count, visualize_distribution

RESULT_DIR = get_results_dir(__file__)


@pytest.mark.parametrize(
    "time",
    [None, np.array(["2011-01-01", "2012-01-01", "2013-01-01"], dtype="datetime64[D]")],
)
def test_plot_zero_count(time: ArrayLike | None) -> None:
    r"""Count exact zeros, excluding negative and missing values."""
    data = [[0, -1, np.nan], [0, 0, np.nan], [1, 2, 3]]
    fig, ax = plt.subplots()
    try:
        assert plot_zero_count(data, time=time, ax=ax) is ax
        (line,) = ax.lines
        assert_array_equal(line.get_xdata(), np.arange(3) if time is None else time)
        assert_array_equal(line.get_ydata(), [1, 2, 0])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("shape", [(0, 2), (3, 0)])
def test_plot_zero_count_empty(shape: tuple[int, int]) -> None:
    r"""Allow empty observations and frames without channels."""
    ax = plot_zero_count(np.empty(shape))
    try:
        assert_array_equal(ax.lines[0].get_ydata(), np.zeros(shape[0], dtype=int))
    finally:
        plt.close()


@pytest.mark.parametrize(
    ("data", "time"),
    [([0, 1], None), ([[0], [1]], [0]), ([[0], [1]], [[0], [1]])],
)
def test_plot_zero_count_invalid(data: ArrayLike, time: ArrayLike | None) -> None:
    r"""Reject ambiguous array shapes and mismatched time coordinates."""
    with pytest.raises(ValueError, match=r"two-dimensional|time coordinate"):
        plot_zero_count(data, time=time)


def test_visualize_distribution() -> None:
    r"""Test on gamma distribution."""
    N = 512
    data = gamma.rvs(2, size=N)

    fig, ax = plt.subplots(figsize=(5, 3))
    visualize_distribution(data, ax=ax, num_bins=50, log=True)
    fig.savefig(RESULT_DIR / "test_visualize_distribution.png", dpi=300)


def test_visualize_distribution_bimodal() -> None:
    r"""Test on perfectly symmetric data."""
    N = 128
    data = np.concatenate([-np.ones(N), +np.ones(N)])

    fig, ax = plt.subplots(figsize=(5, 3))
    visualize_distribution(data, ax=ax, num_bins=50, log=False)
    fig.savefig(RESULT_DIR / "test_visualize_distribution_bimodal.png", dpi=300)
