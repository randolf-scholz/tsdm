r"""Stdlib random sources and sequence-based task partitions."""

from collections.abc import Callable
from functools import partial
from random import Random, getstate
from types import MappingProxyType

import pytest

from tsdm.tasks.utils import (
    is_partition,
    random_partition,
    repeated_random_folds,
    rotating_folds,
)


@pytest.fixture(params=["partition", "rotating", "repeated"])
def partitioner(request: pytest.FixtureRequest) -> Callable[..., object]:
    r"""Exercise each public API using the same collection identifiers."""
    keys = list(range(20))
    match request.param:
        case "partition":
            return partial(random_partition, keys, sizes=(12, 8))
        case "rotating":
            return partial(
                rotating_folds, keys, chunks={"train": 3, "test": 1}, num_folds=4
            )
        case "repeated":
            return partial(
                repeated_random_folds, keys, ratios={"train": 3, "test": 1}, num_folds=4
            )
        case _:
            raise AssertionError(request.param)


def test_seed_and_instance_equivalence(partitioner: Callable[..., object]) -> None:
    before = getstate()
    assert partitioner(rng=37) == partitioner(rng=Random(37))
    assert getstate() == before


def test_existing_rng_advances_and_can_be_replayed(
    partitioner: Callable[..., object],
) -> None:
    rng = Random(37)
    initial = rng.getstate()
    first = partitioner(rng=rng)
    after_first = rng.getstate()
    second = partitioner(rng=rng)
    assert initial != after_first != rng.getstate()
    assert first != second

    rng.setstate(initial)
    assert partitioner(rng=rng) == first
    assert partitioner(rng=rng) == second


@pytest.mark.parametrize("rng", [None, 1.5, "37"])
def test_invalid_random_source(partitioner: Callable[..., object], rng) -> None:
    with pytest.raises(TypeError, match="rng"):
        partitioner(rng=rng)


def test_random_source_is_required(partitioner: Callable[..., object]) -> None:
    with pytest.raises(TypeError, match="rng"):
        partitioner()


def test_exact_sizes_and_tuple_identifiers() -> None:
    keys = [(index, "series") for index in range(7000)]
    parts = random_partition(keys, sizes=(5000, 1000, 1000), rng=42)
    assert tuple(len(part) for part in parts) == (5000, 1000, 1000)
    assert is_partition(parts, union=keys)
    assert parts == random_partition(keys, sizes=(5000, 1000, 1000), rng=42)


def test_largest_remainder_ties_favor_later_parts() -> None:
    parts = random_partition(range(1), ratios=(1, 1), rng=0)
    assert tuple(len(part) for part in parts) == (0, 1)


def test_large_finite_ratios() -> None:
    parts = random_partition(range(5), ratios=(1e308, 1e308), rng=0)
    assert tuple(len(part) for part in parts) == (2, 3)
    assert is_partition(parts, union=range(5))


def test_empty_partitions() -> None:
    assert random_partition([], sizes=(0, 0), rng=0) == [[], []]
    assert random_partition([], ratios=(1, 0), rng=0) == [[], []]
    assert random_partition([], sizes={}, rng=0) == {}
    assert random_partition([], ratios={"train": 1, "test": 0}, rng=0) == {
        "train": [],
        "test": [],
    }


@pytest.mark.parametrize("sizes", [(-1, 3), (1, 0)])
@pytest.mark.parametrize("named", [True, False])
def test_invalid_sizes(sizes, *, named: bool) -> None:
    with pytest.raises(ValueError, match="sizes"):
        random_partition(
            range(2), sizes=dict(enumerate(sizes)) if named else sizes, rng=0
        )


@pytest.mark.parametrize(
    "ratios", [(), (0, 0), (-1, 2), (float("nan"), 1), (float("inf"), 1)]
)
@pytest.mark.parametrize("named", [True, False])
def test_invalid_ratios(ratios, *, named: bool) -> None:
    with pytest.raises(ValueError, match="ratios"):
        random_partition(
            range(2), ratios=dict(enumerate(ratios)) if named else ratios, rng=0
        )


def test_exactly_one_allocation_mode() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        random_partition([1], rng=0)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="Exactly one"):
        random_partition([1], sizes=(1,), ratios=(1,), rng=0)  # type: ignore[call-overload]


@pytest.mark.parametrize("parameter", ["sizes", "ratios"])
def test_named_and_unnamed_partitions(parameter: str) -> None:
    r"""Mapping keys retain identity and order without changing seeded assignments."""
    marker = object()
    spec = MappingProxyType({"train": 3, "training": 3, ("custom", 1): 2, marker: 0})
    items = [[index % 2] for index in range(8)]
    partitioner: Callable[..., object] = random_partition
    named = partitioner(items, **{parameter: spec}, rng=42)
    assert isinstance(named, dict)
    assert all(actual is key for actual, key in zip(named, spec, strict=True))
    assert [len(part) for part in named.values()] == [3, 3, 2, 0]
    for counts in (list(spec.values()), tuple(spec.values())):
        unnamed = partitioner(items, **{parameter: counts}, rng=42)
        assert isinstance(unnamed, list)
        assert list(named.values()) == unnamed
        for named_part, unnamed_part in zip(named.values(), unnamed, strict=True):
            assert all(
                left is right
                for left, right in zip(named_part, unnamed_part, strict=True)
            )
