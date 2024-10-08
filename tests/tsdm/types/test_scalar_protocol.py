r"""Tests for `tsdm.types.scalars`."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch as pt

from tsdm.testing import check_shared_interface
from tsdm.types.scalars import (
    AdditiveScalar,
    BoolScalar,
    ComplexScalar,
    FloatScalar,
    IntScalar,
    OrderedScalar,
)

BASE_SCALARS: dict[object, object] = {
    None         : None,
    bool         : True,
    int          : 1,
    float        : 1.0,
    complex      : 1 + 1j,
    dt.datetime  : dt.datetime(2021, 1, 1),
    dt.timedelta : dt.timedelta(days=1),
    pd.NA        : pd.NA,
    pd.NaT       : pd.NaT,
}  # fmt: skip
r"""Base scalars for testing."""

ORDERED_SCALARS: dict[str, OrderedScalar] = {
    "np_bool"      : np.True_,
    "np_datetime"  : np.datetime64("2021-01-01"),
    "np_float"     : np.float64(1.0),
    "np_int"       : np.int64(1),
    "np_timedelta" : np.timedelta64(1, "D"),
    "pd_timedelta" : pd.Timedelta("1D"),
    "pd_timestamp" : pd.Timestamp("2021-01-01"),
    "pd_nat"       : pd.NaT,
    "pd_na"        : pd.NA,
    "py_string"    : "1",
    "py_bytes"     : b"1",
    "py_bool"      : True,
    "py_tuple"     : (1,),
    "py_datetime"  : dt.datetime(2021, 1, 1),
    "py_float"     : 1.0,
    "py_int"       : 1,
    "py_timedelta" : dt.timedelta(days=1),
    "pt_bool"      : pt.tensor([True], dtype=pt.bool),
    "pt_int"       : pt.tensor(1, dtype=pt.int64),
    "pt_float"     : pt.tensor(1.0, dtype=pt.float64),
}  # fmt: skip
r"""Ordered scalars for testing."""

ADDITIVE_SCALARS: dict[str, AdditiveScalar] = {
    "np_complex"   : np.complex128(1 + 1j),
    "np_float"     : np.float64(1.0),
    "np_int"       : np.int64(1),
    "np_timedelta" : np.timedelta64(1, "D"),
    "pd_timedelta" : pd.Timedelta("1D"),
    "pd_nat"       : pd.NaT,
    "pd_na"        : pd.NA,
    "pt_complex"   : pt.tensor(1 + 1j, dtype=pt.complex128),
    "pt_float"     : pt.tensor(1.0, dtype=pt.float64),
    "pt_int"       : pt.tensor(1, dtype=pt.int64),
    "py_complex"   : 1 + 1j,
    "py_float"     : 1.0,
    "py_int"       : 1,
    "py_timedelta" : dt.timedelta(days=1),
}  # fmt: skip
r"""Additive scalars for testing."""

BOOLEAN_SCALARS: dict[str, BoolScalar] = {
    "np_bool" : np.True_,
    "py_bool" : True,
    "pt_bool" : pt.tensor([True], dtype=pt.bool),
}  # fmt: skip
r"""Boolean scalars for testing."""

FLOAT_SCALARS: dict[str, FloatScalar] = {
    "np_float": np.float64(1.0),
    "py_float": 1.0,
    "pt_float": pt.tensor(1.0, dtype=pt.float64),
}  # fmt: skip
r"""Float scalars for testing."""

INT_SCALARS: dict[str, IntScalar] = {
    "np_int": np.int64(1),
    "py_int": 1,
    "pt_int": pt.tensor(1, dtype=pt.int64),
}  # fmt: skip
r"""Integer scalars for testing."""

COMPLEX_SCALARS: dict[str, ComplexScalar] = {
    "np_complex": np.complex128(1 + 1j),
    "py_complex": 1 + 1j,
    "pt_complex": pt.tensor(1 + 1j, dtype=pt.complex128),
}  # fmt: skip
r"""Complex scalars for testing."""

TEST_TYPED_CASES: dict[type, dict] = {
    BoolScalar     : BOOLEAN_SCALARS,
    ComplexScalar  : COMPLEX_SCALARS,
    FloatScalar    : FLOAT_SCALARS,
    IntScalar      : INT_SCALARS,
    # TimeDelta      : TIMEDELTA_SCALARS,
    # DateTime       : DATETIME_SCALARS,
}  # fmt: skip
r"""Test cases for scalar types."""


@pytest.mark.parametrize("name", BOOLEAN_SCALARS)
def test_boolean_scalar(name: str) -> None:
    value = BOOLEAN_SCALARS[name]
    assert isinstance(value, BoolScalar)
    assert issubclass(value.__class__, BoolScalar)

    # test __bool__
    assert bool(value) == value
    assert isinstance(bool(value), bool)
    # test __and__
    assert value & value == value
    assert isinstance(value & value, BoolScalar)
    # test __or__
    assert value | value == value
    assert isinstance(value | value, BoolScalar)
    # test __xor__
    assert value ^ value != value
    assert isinstance(value ^ value, BoolScalar)


@pytest.mark.parametrize("name", ORDERED_SCALARS)
def test_ordered_scalar(name: str) -> None:
    value = ORDERED_SCALARS[name]
    assert isinstance(value, OrderedScalar)

    # test ==
    assert value == value
    assert isinstance(value == value, BoolScalar)
    # test !=
    assert value == value
    assert isinstance(value != value, BoolScalar)
    # test <=
    assert value <= value
    assert isinstance(value <= value, BoolScalar)
    # test <
    assert not (value < value)
    assert isinstance(value < value, BoolScalar)
    # test >=
    assert value >= value
    assert isinstance(value >= value, BoolScalar)
    # test >
    assert not (value > value)
    assert isinstance(value > value, BoolScalar)


@pytest.mark.parametrize("name", ADDITIVE_SCALARS)
def test_additive_scalar(name: str) -> None:
    value = ADDITIVE_SCALARS[name]
    assert isinstance(value, AdditiveScalar)
    cls = value.__class__

    # test __add__
    assert isinstance(value + value, cls)
    # test __sub__
    assert isinstance(value - value, cls)


@pytest.mark.parametrize("protocol", TEST_TYPED_CASES)
def test_shared_interface(protocol: type) -> None:
    test_cases = TEST_TYPED_CASES[protocol]
    check_shared_interface(test_cases.values(), protocol, raise_on_extra=False)
