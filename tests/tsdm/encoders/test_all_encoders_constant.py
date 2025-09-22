r"""Tests for the ENCODERS module constant."""

from dataclasses import fields

import pytest

from tsdm.encoders import ENCODERS
from tsdm.types.protocols import Dataclass


@pytest.mark.parametrize("name", ENCODERS)
def test_fields_classvar(name: str) -> None:
    r"""Check if the encoder has implemented the `params` property."""
    cls = ENCODERS[name]

    actual_fields = cls.FIELDS
    if issubclass(cls, Dataclass):  # type: ignore[misc]
        expected_fields = {f.name for f in fields(cls)}
        assert actual_fields == expected_fields
