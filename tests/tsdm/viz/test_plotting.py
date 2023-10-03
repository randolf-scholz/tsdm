#!/usr/bin/env python
r"""Test tsdm.viz plotting utilities."""

import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gamma

from tsdm.config import PROJECT
from tsdm.viz import visualize_distribution

RESULT_DIR = PROJECT.TEST_RESULTS_PATH / (PROJECT.TEST_RESULTS_PATH / __file__).stem
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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


def __main__():
    logging.basicConfig(level=logging.INFO)
    test_visualize_distribution()


if __name__ == "__main__":
    __main__()