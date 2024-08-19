r"""Tests for the ENCODERS module constant."""

from inspect import isabstract

import pytest

import tsdm
from tsdm.datasets import DATASETS, BaseDataset


@pytest.mark.parametrize("name", DATASETS)
def test_issubclass(name: str) -> None:
    r"""Check if the encoder has implemented the `params` property."""
    cls = DATASETS[name]
    assert not isabstract(cls), "Class must not be abstract."
    assert issubclass(cls, BaseDataset), "Class must be a subclass of BaseDataset."


def test_dict_complete() -> None:
    r"""Check if all datasets are in the DATASETS constant."""
    names: set[str] = {
        name
        for name, obj in vars(tsdm.datasets).items()
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseDataset)
            and not isabstract(obj)
        )
    }
    if missing := names - DATASETS.keys():
        raise AssertionError(f"Missing datasets: {missing}")


def test_correct_names():
    for name, dataset in DATASETS.items():
        assert (
            name == dataset.__name__
        ), f"Name of dataset {name} does not match the class name {dataset.__name__}"
