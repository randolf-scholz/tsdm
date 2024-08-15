r"""Tests scalar types."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch as pt
from typing_extensions import get_protocol_members

from tsdm.types.protocols import AdditiveScalar, BooleanScalar, OrderedScalar

BASE_SCALARS: dict[object, object] = {
    None         : None,
    bool         : True,
    int          : 1,
    float        : 1.0,
    complex      : 1 + 1j,
    dt.datetime  : dt.datetime(2021, 1, 1),
    dt.timedelta : dt.timedelta(days=1),
    pd.NA        : pd.NA,
}  # fmt: skip

BOOL_SCALAR_TYPES: dict[str, BooleanScalar] = {
    "np_bool" : np.bool_([True]),
    "py_bool" : True,
    "pt_bool" : pt.tensor([True]),
}  # fmt: skip

ORDERED_SCALAR_TYPES: dict[str, OrderedScalar] = {
    "np_bool"      : np.bool_([True]),
    "np_datetime"  : np.datetime64("2021-01-01"),
    "np_float"     : np.float64(1.0),
    "np_int"       : np.int64(1),
    "np_timedelta" : np.timedelta64(1, "D"),
    "pd_timedelta" : pd.Timedelta("1D"),
    "pd_timestamp" : pd.Timestamp("2021-01-01"),
    "py_string"    : "1",
    "py_bytes"     : b"1",
    "py_bool"      : True,
    "py_list"      : [1],
    "py_datetime"  : dt.datetime(2021, 1, 1),
    "py_float"     : 1.0,
    "py_int"       : 1,
    "py_timedelta" : dt.timedelta(days=1),
    "pt_bool"      : pt.tensor([True], dtype=pt.bool),
    "pt_int"       : pt.tensor(1, dtype=pt.int64),
    "pt_float"     : pt.tensor(1.0, dtype=pt.float64),
}  # fmt: skip

ADDITIVE_SCALAR_TYPES: dict[str, AdditiveScalar] = {
    "np_complex"   : np.complex128(1 + 1j),
    "np_float"     : np.float64(1.0),
    "np_int"       : np.int64(1),
    "np_timedelta" : np.timedelta64(1, "D"),
    "pd_timedelta" : pd.Timedelta("1D"),
    "pt_complex"   : pt.tensor(1 + 1j, dtype=pt.complex128),
    "pt_float"     : pt.tensor(1.0, dtype=pt.float64),
    "pt_int"       : pt.tensor(1, dtype=pt.int64),
    "py_complex"   : 1 + 1j,
    "py_float"     : 1.0,
    "py_int"       : 1,
    "py_timedelta" : dt.timedelta(days=1),
}  # fmt: skip


@pytest.mark.parametrize("name", BOOL_SCALAR_TYPES)
def test_boolean_interface(name: str) -> None:
    value = BOOL_SCALAR_TYPES[name]

    # test __bool__
    assert bool(value) == value
    assert isinstance(bool(value), bool)
    # test __and__
    assert value & value == value
    assert isinstance(value & value, BooleanScalar)
    # test __or__
    assert value | value == value
    assert isinstance(value | value, BooleanScalar)
    # test __xor__
    assert value ^ value != value
    assert isinstance(value ^ value, BooleanScalar)


@pytest.mark.parametrize("name", ORDERED_SCALAR_TYPES)
def test_protocol_interface(name: str) -> None:
    value = ORDERED_SCALAR_TYPES[name]

    # test ==
    assert value == value
    assert isinstance(value == value, BooleanScalar)
    # test !=
    assert value == value
    assert isinstance(value != value, BooleanScalar)
    # test <=
    assert value <= value
    assert isinstance(value <= value, BooleanScalar)
    # test <
    assert not (value < value)
    assert isinstance(value < value, BooleanScalar)
    # test >=
    assert value >= value
    assert isinstance(value >= value, BooleanScalar)
    # test >
    assert not (value > value)
    assert isinstance(value > value, BooleanScalar)


@pytest.mark.parametrize("name", ADDITIVE_SCALAR_TYPES)
def test_additive_interface(name: str) -> None:
    value = ADDITIVE_SCALAR_TYPES[name]
    cls = value.__class__

    # test __add__
    assert isinstance(value + value, cls)
    # test __sub__
    assert isinstance(value - value, cls)


def test_shared_interface_bool() -> None:
    shared_attrs = set.intersection(
        *(set(dir(obj)) for obj in BOOL_SCALAR_TYPES.values())
    )
    interface = get_protocol_members(BooleanScalar)

    superfluous = interface - shared_attrs
    assert not superfluous, f"{superfluous=}"
    # assert not (missing := sorted(shared_attrs - interface)), f"{missing=}"


def test_shared_interface_ordered() -> None:
    shared_attrs = set.intersection(
        *(set(dir(obj)) for obj in ORDERED_SCALAR_TYPES.values())
    )
    interface = get_protocol_members(OrderedScalar)

    superfluous = interface - shared_attrs
    assert not superfluous, f"{superfluous=}"
    # assert not (missing := sorted(shared_attrs - interface)), f"{missing=}"


def test_shared_interface_additive() -> None:
    shared_attrs = set.intersection(
        *(set(dir(obj)) for obj in ADDITIVE_SCALAR_TYPES.values())
    )
    interface = get_protocol_members(AdditiveScalar)

    superfluous = interface - shared_attrs
    assert not superfluous, f"{superfluous=}"
    # assert not (missing := shared_attrs - interface), f"{missing=}"
