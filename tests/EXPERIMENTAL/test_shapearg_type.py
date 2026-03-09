r"""Tests for ShapeArg type alias."""

import numpy as np
import pytest

ShapeArg = int | tuple[int, ...]
r"""Type Alias for shape-like objects (note: `ones(shape=None)` creates 0d-array."""


@pytest.mark.parametrize("shape", [0, 1, (), (0,), (1,), (1, 2)], ids=str)
def test_shape_to_tuple(shape: ShapeArg) -> None:
    r"""Test `tsdm.utils.shape_to_tuple`."""
    shape_tuple = shape_to_tuple(shape)
    result = np.ones(shape_tuple)
    reference = np.ones(shape)
    assert type(result) is type(reference)
    assert result.shape == reference.shape
    assert (result == reference).all()


def shape_to_tuple(shape: ShapeArg, /) -> tuple[int, ...]:
    r"""Convert shape to tuple.

    Note:
        - `np.ones(shape=None)` produces a 0d-array (1 element).
        - `np.ones(shape=())`   produces a 0d-array (1 element).
        - `np.ones(shape=k)`    produces a 1d-array (k elements).
    """
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)
