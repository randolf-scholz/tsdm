r"""Test `module.class`."""

import pytest

from tsdm.utils import flatten_dict, last, pairwise_disjoint, replace, unflatten_dict


def test_last():
    """Test `tsdm.utils.last`."""
    # test with Sequence.
    x = [1, 2, 3]
    assert last(x) == 3

    # test with reversible (non-sequence)
    x = {1: 1, 2: 2, 3: 3}
    assert last(x) == 3

    # test with generator
    x = (i for i in range(3))
    assert last(x) == 2


def test_last_empty():
    """Test `tsdm.utils.last` with empty input."""
    with pytest.raises(ValueError, match="Sequence is empty!"):
        last([])

    with pytest.raises(ValueError, match="Reversible is empty!"):
        last({})

    with pytest.raises(ValueError, match="Iterable is empty!"):
        last((i for i in range(0)))


def test_replace():
    """Test `tsdm.utils.replace`."""
    string = "Hello World"
    replacements = {"Hello": "Goodbye", "World": "Earth"}
    assert replace(string, replacements) == "Goodbye Earth"

    string = ""
    assert replace(string, replacements) == ""


def test_flatten_dict():
    """Test `tsdm.utils.flatten_dict`."""
    d = {}
    assert flatten_dict(d) == {}

    d = {"a": {"b": {"c": 1}, "d": {"e": 2}, "f": 3}}
    assert flatten_dict(d) == {"a.b.c": 1, "a.d.e": 2, "a.f": 3}


def test_unflatten_dict():
    """Test `tsdm.utils.unflatten_dict`."""
    d = {}
    assert unflatten_dict(d) == {}

    d = {"a.b.c": 1, "a.d.e": 2, "a.f": 3}
    assert unflatten_dict(d) == {"a": {"b": {"c": 1}, "d": {"e": 2}, "f": 3}}


def test_pairwise_disjoint():
    """Test `tsdm.utils.pairwise_disjoint`."""
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
