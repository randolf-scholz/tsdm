#!/usr/bin/env python
r"""Test PhysioNet 2012."""

import logging

import pytest
from pandas import DataFrame

from tsdm.datasets import Physionet2012

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@pytest.mark.skip("fails - PhysioNet2012 needs to be fixed")
def test_physionet_2012():
    """Test the PhysioNet 2012 dataset."""
    LOGGER = __logger__.getChild(Physionet2012.__name__)
    LOGGER.info("Testing.")
    dataset = Physionet2012().dataset
    metadata, timeseries = dataset["A"]
    assert isinstance(metadata, DataFrame)
    assert isinstance(timeseries, DataFrame)


def _main() -> None:
    test_physionet_2012()


if __name__ == "__main__":
    _main()
