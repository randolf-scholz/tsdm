r"""Tests for the ENCODERS module constant."""

from collections.abc import Mapping
from dataclasses import fields

import pytest

from tsdm.encoders import ENCODERS, pandas, polars, torch
from tsdm.types.dataclass import Dataclass


@pytest.mark.parametrize("name", ENCODERS)
def test_fields_classvar(name: str) -> None:
    r"""Check if the encoder has implemented the `params` property."""
    cls = ENCODERS[name]

    actual_fields = cls.FIELDS
    if issubclass(cls, Dataclass):  # type: ignore
        expected_fields = {f.name for f in fields(cls)}
        assert actual_fields == expected_fields


@pytest.mark.parametrize(
    ("namespace", "registry"),
    [("pandas", pandas.ENCODERS), ("polars", polars.ENCODERS)],
)
def test_backend_registry_is_namespaced(
    namespace: str, registry: Mapping[str, type]
) -> None:
    r"""Check that backend registries use qualified keys in the root registry."""
    for name, cls in registry.items():
        assert ENCODERS[f"{namespace}.{name}"] is cls


def test_root_registry_contains_only_canonical_keys() -> None:
    r"""Check that the root registry has no unqualified backend aliases."""
    expected = {f"pandas.{name}" for name in pandas.ENCODERS}
    expected.update(f"polars.{name}" for name in polars.ENCODERS)
    expected.update(f"torch.{name}" for name in torch.ENCODERS)
    qualified = {name for name in ENCODERS if "." in name}

    assert qualified == expected
    assert not set(pandas.ENCODERS) & ENCODERS.keys()
    assert not set(polars.ENCODERS) & ENCODERS.keys()
