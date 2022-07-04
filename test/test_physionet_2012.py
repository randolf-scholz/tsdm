"""Test PhysioNet 2012."""

import pytest


@pytest.mark.skip("Heavy test")
def test_physionet_2012():
    """Test PhysioNet 2012."""
    import tsdm

    dataset = tsdm.datasets.Physionet2012().dataset
    metadata, series = dataset["A"]

    print(metadata)
    print(series)
