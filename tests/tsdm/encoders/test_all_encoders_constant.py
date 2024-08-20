r"""Tests for the ENCODERS module constant."""

import pytest

from tsdm.encoders import ENCODERS, BaseEncoder
from tsdm.types.protocols import Dataclass


@pytest.mark.parametrize("name", ENCODERS)
def test_params_property(name: str) -> None:
    r"""Check if the encoder has implemented the `params` property."""
    cls = ENCODERS[name]
    # check params is implemented if subclass is not a dataclass.
    if cls.params is BaseEncoder.params and not issubclass(cls, Dataclass):  # type: ignore[misc]
        raise NotImplementedError(f"Class {cls} must be implement `params` property.")
