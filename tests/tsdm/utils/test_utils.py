r"""Test `module.class`."""

from typing import Any

import numpy as np
import pytest
import torch
from torch import jit

from tsdm.types.aliases import Axis, DimArg
from tsdm.utils import (
    flatten_dict,
    last,
    normalize_axes,
    normalize_dimarg,
    replace,
    unflatten_dict,
)


@pytest.mark.parametrize("dims", [None, 0, 1, [], [0], [-1], [-1, -2]], ids=str)
def test_dims_to_list(dims: DimArg) -> None:
    r"""Test `tsdm.utils.dims_to_list`."""
    x = torch.randn(4, 2, 2, 1)

    # test
    dims_list: list[int] = normalize_dimarg(dims, ndim=x.ndim)
    result = x.mean(dims_list)
    reference = x.mean(dim=dims)
    assert type(result) is type(reference)
    assert result.shape == reference.shape
    assert (result == reference).all()

    # test with jit.script
    if dims == []:
        pytest.xfail("JIT compiler cannot determine type of empty list.")
    f = jit.script(normalize_dimarg)
    dims_list = f(dims, ndim=x.ndim)
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


def test_last() -> None:
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


def test_last_empty() -> None:
    r"""Test `tsdm.utils.last` with empty input."""
    with pytest.raises(ValueError, match="Sequence is empty!"):
        last([])

    with pytest.raises(ValueError, match="Reversible is empty!"):
        last({})

    with pytest.raises(ValueError, match="Iterable is empty!"):
        last(i for i in range(0))


def test_replace() -> None:
    r"""Test `tsdm.utils.replace`."""
    string = "Hello World"
    replacements = {"Hello": "Goodbye", "World": "Earth"}
    assert replace(string, replacements) == "Goodbye Earth"


@pytest.mark.parametrize(
    ("d", "kwargs", "expected"),
    [
        pytest.param(
            {"a": {"b": 1, "c": 2}},
            {},
            {"a.b": 1, "a.c": 2},
            id="dot_flat_simple",
        ),
        pytest.param(
            {"a": {"b": {"x": 2}, "c": 2}},
            {},
            {"a.b.x": 2, "a.c": 2},
            id="dot_flat_nested",
        ),
        pytest.param(
            {"a": {"b": 1, "c": 2}},
            {"join_fn": tuple, "split_fn": lambda x: x},
            {("a", "b"): 1, ("a", "c"): 2},
            id="tuple_flat_simple",
        ),
        pytest.param(
            {"a": {"b": {"x": 2}, "c": 2}},
            {"join_fn": tuple, "split_fn": lambda x: x},
            {("a", "b", "x"): 2, ("a", "c"): 2},
            id="tuple_flat_nested",
        ),
        pytest.param(
            {"a": {"i": {"x": 0}, "b": {"y": 1}}},
            {},
            {"a.i.x": 0, "a.b.y": 1},
            id="dot_partial_default",
        ),
        pytest.param(
            {"a": {"i": {"x": 0}, "b": {"y": 1}}},
            {"recursive": 2},
            {"a.i": {"x": 0}, "a.b": {"y": 1}},
            id="dot_partial_recursive_2",
        ),
        pytest.param(
            {"a": {"i": {"x": 0}, "b": {"y": 1}}},
            {"recursive": 1},
            {"a": {"i": {"x": 0}, "b": {"y": 1}}},
            id="dot_partial_recursive_1",
        ),
    ],
)
def test_flatten_dict_doctests(d: dict, kwargs: Any, expected: dict) -> None:
    result = flatten_dict(d, **kwargs)
    assert result == expected


def test_flatten_dict() -> None:
    r"""Test `tsdm.utils.flatten_dict`."""
    empty_dict: dict = {}
    assert flatten_dict(empty_dict) == {}

    d = {
        "a": {
            "b": {"c": 1},
            "d": {"e": 2},
            "f": 3,
        },
        "g": 4,
    }
    tuple_target = {("a", "b", "c"): 1, ("a", "d", "e"): 2, ("a", "f"): 3, ("g",): 4}
    tuple_result = flatten_dict(d, join_fn=tuple, split_fn=lambda x: x)
    assert tuple_result == tuple_target

    dot_target = {"a.b.c": 1, "a.d.e": 2, "a.f": 3, "g": 4}
    dot_result = flatten_dict(d, join_fn=".".join, split_fn=lambda x: x.split("."))
    assert dot_result == dot_target


def test_unflatten_dict() -> None:
    r"""Test `tsdm.utils.unflatten_dict`."""
    empty_dict: dict = {}
    assert unflatten_dict(empty_dict) == {}

    result = {
        "a": {
            "b": {"c": 1},
            "d": {"e": 2},
            "f": 3,
        },
        "g": 4,
    }

    tup = {("a", "b", "c"): 1, ("a", "d", "e"): 2, ("a", "f"): 3, ("g",): 4}
    tuple_result = unflatten_dict(tup, join_fn=tuple, split_fn=lambda x: x)
    assert tuple_result == result

    dot = {"a.b.c": 1, "a.d.e": 2, "a.f": 3, "g": 4}
    dot_result = unflatten_dict(dot, join_fn=".".join, split_fn=lambda x: x.split("."))
    assert dot_result == result
