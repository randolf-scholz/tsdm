r"""Test PhysioNet 2012."""

import logging

import pytest
from pandas import DataFrame

from tsdm.datasets import PhysioNet2012

__logger__ = logging.getLogger(__name__)


@pytest.mark.skip("fails - PhysioNet2012 needs to be fixed")
def test_physionet_2012() -> None:
    """Test the PhysioNet 2012 dataset."""
    LOGGER = __logger__.getChild(PhysioNet2012.__name__)
    LOGGER.info("Testing.")
    ds = PhysioNet2012()

    assert isinstance(ds.metadata, DataFrame)
    assert isinstance(ds.timeseries, DataFrame)
