r"""Test tsdm.viz plotting utilities."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gamma

from tsdm.config import PROJECT
from tsdm.viz import visualize_distribution

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


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
