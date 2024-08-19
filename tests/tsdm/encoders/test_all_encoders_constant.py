r"""Tests for the ENCODERS module constant."""

from inspect import isabstract

import pytest

import tsdm
from tsdm.encoders import ENCODERS, BaseEncoder
from tsdm.types.protocols import Dataclass


@pytest.mark.parametrize("name", ENCODERS)
def test_params_property(name: str) -> None:
    r"""Check if the encoder has implemented the `params` property."""
    cls = ENCODERS[name]
    assert not isabstract(cls), "Class must not be abstract."
    assert issubclass(cls, BaseEncoder), "Class must be a subclass of BaseEncoder."

    # check params is implemented if subclass is not a dataclass.
    if cls.params is BaseEncoder.params and not issubclass(cls, Dataclass):  # type: ignore[misc]
        raise NotImplementedError(f"Class {cls} must be implement `params` property.")


def test_dict_complete() -> None:
    r"""Check if all encoders are in the ENCODERS constant."""
    names: set[str] = {
        name
        for name, obj in vars(tsdm.encoders).items()
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseEncoder)
            and not isabstract(obj)
        )
    }
    if missing := names - ENCODERS.keys():
        raise AssertionError(f"Missing encoders: {missing}")


def test_correct_names():
    for name, encoder in ENCODERS.items():
        assert (
            name == encoder.__name__
        ), f"Name of encoder {name} does not match the class name {encoder.__name__}"
