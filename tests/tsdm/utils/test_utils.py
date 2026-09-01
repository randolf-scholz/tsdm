r"""Test `module.class`."""

from typing import Any

import pytest

from tsdm.utils import flatten_dict, unflatten_dict


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
    tuple_result = unflatten_dict(tup, join_fn=tuple, split_fn=lambda x: x)  # pyrefly: ignore[no-matching-overload]
    assert tuple_result == result

    dot = {"a.b.c": 1, "a.d.e": 2, "a.f": 3, "g": 4}
    dot_result = unflatten_dict(dot, join_fn=".".join, split_fn=lambda x: x.split("."))
    assert dot_result == result
