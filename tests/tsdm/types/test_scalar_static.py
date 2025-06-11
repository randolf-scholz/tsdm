r"""Tests for `tsdm.types.scalars`."""

import datetime as dt

import numpy as np
import pandas as pd
import torch as pt

from tsdm.types.scalars import (
    BoolScalar,
    ComplexScalar,
    DurationScalar,
    FloatScalar,
    IntScalar,
    TimestampScalar,
)


def test_float_scalar() -> None:
    _1: FloatScalar = np.floating()  # type: ignore[assignment]


def test_timestamp_assignable() -> None:
    # numpy
    _np_0: TimestampScalar[np.int64] = np.int64(0)
    _np_1: TimestampScalar[np.float64] = np.float64(1)
    # FIXME: https://github.com/numpy/numpy/issues/28257
    _np_2: TimestampScalar[np.timedelta64] = np.datetime64("2021-01-01")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _np_3: TimestampScalar[dt.timedelta] = np.datetime64("2021-01-01")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # python
    _py_1: TimestampScalar[dt.timedelta] = dt.datetime(2021, 1, 1)
    _py_2: TimestampScalar[int] = int(3)
    _py_3: TimestampScalar[float] = float(3.0)
    # pandas
    _pd_1: TimestampScalar[dt.timedelta] = pd.Timestamp("2021-01-01")
    _pd_2: TimestampScalar[pd.Timedelta] = pd.Timestamp("2021-01-01")


def test_timedelta_assignable() -> None:
    # numpy
    _np_0: DurationScalar = np.int64(0)
    _np_1: DurationScalar = np.float64(1)
    # FIXME: https://github.com/numpy/numpy/issues/28257
    _np_2: DurationScalar = np.timedelta64(1, "D")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _np_3: DurationScalar = np.timedelta64(1, "D")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # python
    _py_1: DurationScalar = dt.timedelta(days=1)
    _py_2: DurationScalar = int(3)
    _py_3: DurationScalar = float(3.0)
    # pandas
    _pd_1: DurationScalar = pd.Timedelta(days=1)


def test_boolean_assignable() -> None:
    # numpy
    _np_0: BoolScalar = np.bool_(bool(1234))
    _np_1: BoolScalar = np.True_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _np_2: BoolScalar = np.False_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # python
    _py_0: BoolScalar = bool(1234)
    _py_2: BoolScalar = True
    _py_3: BoolScalar = False
    # pytorch
    _pt_0: BoolScalar = pt.tensor([True], dtype=pt.bool)


def test_int_assignable() -> None:
    # numpy
    _np_0: IntScalar = np.int64(1234)  # type: ignore[assignment]
    # python
    _py_0: IntScalar = int(1234)  # type: ignore[assignment]
    _py_1: IntScalar = 0  # type: ignore[assignment]
    # pytorch
    _pt_0: IntScalar = pt.tensor([1234], dtype=pt.int64)


def test_float_assignable() -> None:
    # numpy
    _np_0: FloatScalar = np.float64(1234.0)
    # python
    _py_0: FloatScalar = float(1234.0)
    _py_1: FloatScalar = 0.0
    # pytorch
    _pt_0: FloatScalar = pt.tensor([1234.0], dtype=pt.float64)


def test_complex_assignable() -> None:
    # numpy
    _np_0: ComplexScalar = np.complex128(1 + 2j)
    # python
    _py_0: ComplexScalar = complex(1, 2)
    _py_1: ComplexScalar = 0 + 0j
    # pytorch
    _pt_0: ComplexScalar = pt.tensor([1 + 2j], dtype=pt.complex128)
