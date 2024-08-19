r"""Test PhysioNet 2012."""

from pandas import DataFrame

from tsdm.datasets import PhysioNet2012


def test_physionet_2012() -> None:
    r"""Test the PhysioNet 2012 dataset."""
    ds = PhysioNet2012()

    assert isinstance(ds.static_covariates, DataFrame)
    assert isinstance(ds.timeseries, DataFrame)
