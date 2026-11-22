r"""Stdlib random sources and sequence-based task partitions."""

from collections.abc import Callable
from functools import partial
from math import nextafter
from random import Random, getstate
from types import MappingProxyType

import pytest

from tsdm.stats.distances import total_variation_distance
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


@pytest.fixture(params=["partition", "rotating", "repeated"])
def ratio_partitioner(request: pytest.FixtureRequest) -> Callable[..., object]:
    r"""Exercise the same relative targets and error policy through each API."""
    match request.param:
        case "partition":
            return partial(random_partition, ratios=[1, 1], rng=0)
        case "rotating":
            return partial(rotating_folds, chunks=[1, 1], num_folds=2, rng=0)
        case "repeated":
            return partial(repeated_random_folds, ratios=[1, 1], num_folds=2, rng=0)
        case _:
            raise AssertionError(request.param)


def test_error_check_disabled_by_default(
    ratio_partitioner: Callable[..., object],
) -> None:
    r"""Default and explicit None both permit large discrepancies without checking."""
    result = ratio_partitioner(range(1))
    assert result
    assert result == ratio_partitioner(range(1), max_error=None)


def test_max_error_rejects_rounding(
    ratio_partitioner: Callable[..., object],
) -> None:
    r"""One occurrence cannot meet equal targets within an explicit five percent."""
    with pytest.raises(
        RuntimeError, match=r"total variation distance 0.5.*max_error=0.05"
    ):
        ratio_partitioner(range(1), max_error=0.05)


def test_within_max_error(ratio_partitioner: Callable[..., object]) -> None:
    r"""Normalized proportions allow small rounding errors, despite unequal counts."""
    assert ratio_partitioner(range(21), max_error=0.05)


def test_max_error_boundary(ratio_partitioner: Callable[..., object]) -> None:
    r"""A threshold equal to TV is accepted; a smaller threshold rejects the result."""
    assert ratio_partitioner(range(1), max_error=0.5)
    with pytest.raises(RuntimeError, match="total variation distance"):
        ratio_partitioner(range(1), max_error=0.499)


def test_zero_max_error(ratio_partitioner: Callable[..., object]) -> None:
    assert ratio_partitioner(range(8), max_error=0)
    with pytest.raises(RuntimeError, match="max_error=0"):
        ratio_partitioner(range(1), max_error=0)


def test_empty_inputs_without_error_check(
    ratio_partitioner: Callable[..., object],
) -> None:
    r"""Disabling the check returns empty splits without evaluating a distribution."""
    result = ratio_partitioner([])
    assert result in (([], []), [([], []), ([], [])])
    assert result == ratio_partitioner([], max_error=None)


@pytest.mark.parametrize("max_error", [0.0, 1.0])
def test_empty_inputs_have_no_distribution(
    ratio_partitioner: Callable[..., object],
    max_error: float,
) -> None:
    r"""The distance function rejects zero total weight for empty allocations."""
    with pytest.raises(ValueError, match="positive total weight"):
        ratio_partitioner([], max_error=max_error)


@pytest.mark.parametrize(
    "max_error", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_max_error(
    ratio_partitioner: Callable[..., object], max_error: float
) -> None:
    with pytest.raises(ValueError, match="max_error"):
        ratio_partitioner(range(8), max_error=max_error)


@pytest.mark.parametrize(
    ("function", "parameter", "options"),
    [
        (random_partition, "ratios", {}),
        (rotating_folds, "chunks", {"num_folds": 4}),
        (repeated_random_folds, "ratios", {"num_folds": 4}),
    ],
)
@pytest.mark.parametrize("named", [True, False])
def test_computed_threshold_is_compared_strictly(
    function, parameter: str, *, options, named: bool
) -> None:
    r"""The computed distance passes, but even the next smaller threshold fails."""
    spec = {"train": 3, "test": 1} if named else [3, 1]
    result = function(range(10), **{parameter: spec}, **options, rng=0)
    partitions = result if isinstance(result, list) else [result]
    counts = [
        [len(split) for split in (part.values() if isinstance(part, dict) else part)]
        for part in partitions
    ]
    error = total_variation_distance(counts, [3, 1]).max().item()
    assert function(range(10), **{parameter: spec}, **options, rng=0, max_error=error)
    with pytest.raises(RuntimeError, match="total variation distance"):
        function(
            range(10),
            **{parameter: spec},
            **options,
            rng=0,
            max_error=nextafter(error, 0),
        )


@pytest.mark.parametrize("named", [True, False])
def test_explicit_sizes_skip_error_check(*, named: bool) -> None:
    sizes = {"train": 9, "test": 1} if named else [9, 1]
    result = random_partition(range(10), sizes=sizes, rng=0, max_error=0)
    parts = result.values() if isinstance(result, dict) else result
    assert [len(part) for part in parts] == [9, 1]


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
    assert random_partition([], sizes=(0, 0), rng=0) == ([], [])
    assert random_partition([], sizes={}, rng=0) == {}
    assert random_partition([], sizes={}, rng=0, max_error=0) == {}
    assert random_partition([], ratios=(1, 0), rng=0) == ([], [])
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
    named = (
        random_partition(items, sizes=spec, rng=42)
        if parameter == "sizes"
        else random_partition(items, ratios=spec, rng=42)
    )
    assert isinstance(named, dict)
    assert all(actual is key for actual, key in zip(named, spec, strict=True))
    assert [len(part) for part in named.values()] == [3, 3, 2, 0]
    for counts in (list(spec.values()), tuple(spec.values())):
        unnamed = (
            random_partition(items, sizes=counts, rng=42)
            if parameter == "sizes"
            else random_partition(items, ratios=counts, rng=42)
        )
        assert isinstance(unnamed, tuple)
        assert tuple(named.values()) == unnamed
        for named_part, unnamed_part in zip(named.values(), unnamed, strict=True):
            assert all(
                left is right
                for left, right in zip(named_part, unnamed_part, strict=True)
            )
