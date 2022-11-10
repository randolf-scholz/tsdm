#!/usr/bin/env python
r"""Test PhysioNet 2012."""

import logging

from pandas import DataFrame
from pytest import mark

from tsdm.datasets import Physionet2012

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@mark.skip("Heavy test")
def test_physionet_2012():
    __logger__.info("Testing %s.", Physionet2012)
    dataset = Physionet2012().dataset
    metadata, timeseries = dataset["A"]
    assert isinstance(metadata, DataFrame)
    assert isinstance(timeseries, DataFrame)


def _main() -> None:
    test_physionet_2012()


if __name__ == "__main__":
    _main()
