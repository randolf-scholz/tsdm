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

    # check params is implemented if subclass is not a dataclass.
    if cls.params is BaseEncoder.params and not issubclass(cls, Dataclass):  # type: ignore[misc]
        raise NotImplementedError(f"Class {cls} must be implement `params` property.")


def test_encoders_dict_complete() -> None:
    counter = 0
    for name in dir(tsdm.encoders):
        obj = getattr(tsdm.encoders, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseEncoder)
            and not isabstract(obj)
        ):
            assert name in ENCODERS, f"{name} not in ENCODERS"
            counter += 1

    # make sure we have all encoders
    assert counter == len(ENCODERS), f"Missing encoders: {counter}"
