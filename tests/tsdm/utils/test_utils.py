r"""Test `module.class`."""

import numpy as np
import pytest
import torch
from torch import jit

from tsdm.types.aliases import Axis, Dims, Shape, Size
from tsdm.utils import (
    dims_to_list,
    flatten_dict,
    last,
    normalize_axes,
    pairwise_disjoint,
    replace,
    shape_to_tuple,
    size_to_tuple,
    unflatten_dict,
)


@pytest.mark.parametrize("dims", [None, 0, 1, [], [0], [-1], [-1, -2]], ids=str)
def test_dims_to_list(dims: Dims) -> None:
    r"""Test `tsdm.utils.dims_to_list`."""
    x = torch.randn(4, 2, 2, 1)

    # test
    dims_list: list[int] = dims_to_list(dims, ndim=x.ndim)
    result = x.mean(dims_list)
    reference = x.mean(dim=dims)
    assert type(result) is type(reference)
    assert result.shape == reference.shape
    assert (result == reference).all()

    # test with jit.script
    if dims == []:
        pytest.xfail("JIT compiler cannot determine type of empty list.")
    f = jit.script(dims_to_list)
    dims_list = f(dims, ndim=x.ndim)  # pyright: ignore[reportCallIssue, reportAssignmentType]
    result = x.mean(dims_list)
    reference = x.mean(dim=dims)
    assert type(result) is type(reference)
    assert result.shape == reference.shape
    assert (result == reference).all()


@pytest.mark.parametrize(
    "axis", [None, 0, 1, (), (0,), (-1,), (-1, -2), (0, 1, 2, 3)], ids=str
)
def test_axes_to_tuple(axis: Axis) -> None:
    r"""Test `tsdm.utils.axes_to_tuple`."""
    rng = np.random.default_rng()
    x = rng.uniform(size=(4, 2, 2, 1))

    axes_tuple = normalize_axes(axis, ndim=x.ndim)
    result = np.mean(x, axis=axes_tuple)
    reference = np.mean(x, axis=axis)
    assert type(result) is type(reference)
    assert result.shape == reference.shape
    assert (result == reference).all()


@pytest.mark.parametrize("shape", [0, 1, (), (0,), (1,), (1, 2)], ids=str)
def test_shape_to_tuple(shape: Shape) -> None:
    r"""Test `tsdm.utils.shape_to_tuple`."""
    shape_tuple = shape_to_tuple(shape)
    result = np.ones(shape_tuple)
    reference = np.ones(shape)
    assert type(result) is type(reference)
    assert result.shape == reference.shape
    assert (result == reference).all()


@pytest.mark.parametrize("size", [0, 1, (), (0,), (1,), (1, 2)], ids=str)
def test_size_to_tuple(size: Size) -> None:
    r"""Test `tsdm.utils.size_to_tuple`."""
    sizes_tuple = size_to_tuple(size)
    rng = np.random.default_rng(42)
    result = rng.uniform(size=sizes_tuple)
    rng = np.random.default_rng(42)
    reference = rng.uniform(size=size)
    assert type(result) is type(reference)
    assert result.shape == reference.shape
    assert (result == reference).all()


def test_last():
    r"""Test `tsdm.utils.last`."""
    # test with Sequence.
    seq = [1, 2, 3]
    assert last(seq) == 3

    # test with dictionary
    mapping = {1: 1, 2: 2, 3: 3}
    assert last(mapping) == 3

    # test with generator
    gen = (i for i in range(3))
    assert last(gen) == 2


def test_last_empty():
    r"""Test `tsdm.utils.last` with empty input."""
    with pytest.raises(ValueError, match="Sequence is empty!"):
        last([])

    with pytest.raises(ValueError, match="Reversible is empty!"):
        last({})

    with pytest.raises(ValueError, match="Iterable is empty!"):
        last(i for i in range(0))


def test_replace():
    r"""Test `tsdm.utils.replace`."""
    string = "Hello World"
    replacements = {"Hello": "Goodbye", "World": "Earth"}
    assert replace(string, replacements) == "Goodbye Earth"


def test_flatten_dict():
    r"""Test `tsdm.utils.flatten_dict`."""
    d = {"a": {"b": {"c": 1}, "d": {"e": 2}, "f": 3}}
    assert flatten_dict(d) == {"a.b.c": 1, "a.d.e": 2, "a.f": 3}

    d = {}
    assert flatten_dict(d) == {}


def test_unflatten_dict():
    r"""Test `tsdm.utils.unflatten_dict`."""
    d = {"a.b.c": 1, "a.d.e": 2, "a.f": 3}
    assert unflatten_dict(d) == {"a": {"b": {"c": 1}, "d": {"e": 2}, "f": 3}}

    d = {}
    assert unflatten_dict(d) == {}


def test_pairwise_disjoint():
    r"""Test `tsdm.utils.pairwise_disjoint`."""
    sets: list[set[int]] = []
    assert pairwise_disjoint(sets) is True

    sets = [{1, 2}, {3, 4}]
    assert pairwise_disjoint(sets) is True

    sets = [{1, 2}, {2, 3}]
    assert pairwise_disjoint(sets) is False

    sets = [{1, 2}, {2, 3}, {3, 4}]
    assert pairwise_disjoint(sets) is False

    sets = [{1, 2}, {3, 4}, {5, 6}]
    assert pairwise_disjoint(sets) is True
