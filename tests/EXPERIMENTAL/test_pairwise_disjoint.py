r"""Tests for pairwise_disjoint utility."""

from collections.abc import Iterable


def pairwise_disjoint(sets: Iterable[set], /) -> bool:
    r"""Check if sets are pairwise disjoint."""
    union = set().union(*sets)
    return len(union) == sum(len(s) for s in sets)


def test_pairwise_disjoint() -> None:
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
