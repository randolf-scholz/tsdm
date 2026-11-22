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
    random_partitions,
    rotating_folds,
)


@pytest.fixture(params=["partition", "rotating", "repeated", "repeated_sizes"])
def partitioner(request: pytest.FixtureRequest) -> Callable[..., object]:
    r"""Exercise each public API using the same collection identifiers."""
    keys = list(range(20))
    match request.param:
        case "partition":
            return partial(random_partition, keys, sizes=(12, 8))
        case "rotating":
            return partial(rotating_folds, keys, chunks={"train": 3, "test": 1}, num=4)
        case "repeated":
            return partial(
                random_partitions, keys, ratios={"train": 3, "test": 1}, num=4
            )
        case "repeated_sizes":
            return partial(
                random_partitions, keys, sizes={"train": 15, "test": 5}, num=4
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
            return partial(rotating_folds, chunks=[1, 1], num=2, rng=0)
        case "repeated":
            return partial(random_partitions, ratios=[1, 1], num=2, rng=0)
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
        (rotating_folds, "chunks", {"num": 4}),
        (random_partitions, "ratios", {"num": 4}),
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
@pytest.mark.parametrize("repeated", [True, False])
def test_explicit_sizes_skip_error_check(*, named: bool, repeated: bool) -> None:
    sizes = {"train": 9, "test": 1} if named else [9, 1]
    partitions = (
        random_partitions(range(10), num=2, sizes=sizes, rng=0, max_error=0)
        if repeated
        else [random_partition(range(10), sizes=sizes, rng=0, max_error=0)]
    )
    for partition in partitions:
        parts = partition.values() if isinstance(partition, dict) else partition
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


def test_largest_remainder_ties_favor_earlier_parts() -> None:
    parts = random_partition(range(1), ratios=(1, 1), rng=0)
    assert tuple(len(part) for part in parts) == (1, 0)
    parts = random_partition(range(5), ratios=(1, 1, 1), rng=0)
    assert tuple(len(part) for part in parts) == (2, 2, 1)


def test_large_finite_ratios() -> None:
    parts = random_partition(range(5), ratios=(1e308, 1e308), rng=0)
    assert tuple(len(part) for part in parts) == (3, 2)
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


@pytest.mark.parametrize(
    "function", [random_partition, partial(random_partitions, num=2)]
)
def test_exactly_one_allocation_mode(function) -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        function([1], rng=0)
    with pytest.raises(ValueError, match="Exactly one"):
        function([1], sizes=(1,), ratios=(1,), rng=0)


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


@pytest.mark.parametrize("parameter", ["sizes", "ratios"])
@pytest.mark.parametrize("named", [True, False])
def test_group_allocation(parameter: str, *, named: bool) -> None:
    r"""Group counts determine assignments regardless of the number of samples."""
    labels = [(1, "red"), (2, "red"), (1, "red"), (3, "blue"), (1, "red")]
    items = [[index] for index in range(len(labels))]
    items[-1] = items[0]
    unique_labels = list(dict.fromkeys(labels))
    spec = {"train": 1, ("test", 0): 2, "empty": 0} if named else [1, 2, 0]
    options = {parameter: spec}
    parts = random_partition(items, **options, group_labels=labels, rng=42)
    expected = random_partition(unique_labels, **options, rng=42)
    actual_parts = list(parts.values()) if isinstance(parts, dict) else parts
    expected_parts = list(expected.values()) if isinstance(expected, dict) else expected
    if named:
        assert list(parts) == list(spec)
    for actual, expected_labels in zip(actual_parts, expected_parts, strict=True):
        expected_items = [
            item
            for item, item_label in zip(items, labels, strict=True)
            if item_label in expected_labels
        ]
        assert len(actual) == len(expected_items)
        assert all(
            left is right for left, right in zip(actual, expected_items, strict=True)
        )
    assert len(actual_parts[-1]) == 0
    assert parts == random_partition(items, **options, group_labels=labels, rng=42)


@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
@pytest.mark.parametrize("parameter", ["sizes", "ratios"])
def test_random_and_repeated_partitions_agree(
    parameter: str, *, named: bool, grouped: bool
) -> None:
    r"""Successive draws agree on rounding, membership, input order, and RNG state."""
    items = [7, 0, 6, 1, 5, 2, 4, 3]
    labels = [0, 1, 0, 2, 3, 1, 4, 4] if grouped else None
    counts = [2, 2, 1, 0] if grouped else [3, 3, 2, 0]
    values = counts if parameter == "sizes" else [1, 1, 1, 0]
    spec = (
        dict(zip(["train", "valid", "test", "empty"], values, strict=True))
        if named
        else values
    )
    options = {parameter: spec}
    rng = Random(37)
    reference_rng = Random(37)
    expected = random_partitions(
        items, num=3, **options, group_labels=labels, rng=reference_rng
    )
    actual = [
        random_partition(items, **options, group_labels=labels, rng=rng)
        for _ in range(3)
    ]
    assert actual == expected
    assert rng.getstate() == reference_rng.getstate()
    for partition in actual:
        parts = partition.values() if isinstance(partition, dict) else partition
        for part in parts:
            assert part == [item for item in items if item in part]


@pytest.mark.parametrize("parameter", ["sizes", "ratios"])
def test_unique_labels_match_ungrouped_partition(parameter: str) -> None:
    r"""Each occurrence is its own group by default, including duplicate values."""
    items = [[0], [1], [0], [1]]
    options = {parameter: [2, 2]}
    assert random_partition(items, **options, rng=37) == random_partition(
        items, **options, group_labels=["d", "b", "a", "c"], rng=37
    )


def test_group_sizes_count_groups() -> None:
    r"""Explicit sizes count groups, never sample counts."""
    assert random_partition(
        range(4), sizes=[1, 0], group_labels=["a"] * 4, rng=0, max_error=0
    ) == ([0, 1, 2, 3], [])
    with pytest.raises(ValueError, match="number of groups"):
        random_partition(range(4), sizes=[4, 0], group_labels=["a"] * 4, rng=0)


def test_group_sample_sizes_do_not_bias_assignment() -> None:
    r"""Either unequal group can be the test set, even if it misses sample targets."""
    labels = ["a"] * 9 + ["b"]
    test_sets = {
        tuple(
            random_partition(range(10), ratios=[1, 1], group_labels=labels, rng=seed)[1]
        )
        for seed in range(16)
    }
    assert test_sets == {tuple(range(9)), (9,)}
    assert (
        random_partition(range(10), ratios=[9, 1], group_labels=labels, rng=0)[1] == []
    )


def test_group_error_checks_samples_without_retrying() -> None:
    r"""Equal group counts can violate sample targets; failure consumes just one draw."""
    options = {"ratios": [1, 1], "group_labels": ["a"] * 9 + ["b"]}
    rng = Random(37)
    reference_rng = Random(37)
    expected = random_partition(range(10), **options, rng=reference_rng)
    assert random_partition(range(10), **options, rng=37, max_error=0.41) == expected
    with pytest.raises(RuntimeError, match="total variation distance"):
        random_partition(range(10), **options, rng=rng, max_error=0.39)
    assert rng.getstate() == reference_rng.getstate()


@pytest.mark.parametrize("labels", [[], ["a", "b"]])
def test_partition_invalid_group_lengths(labels) -> None:
    with pytest.raises(ValueError, match=r"group_labels.*length"):
        random_partition([0], ratios=[1, 1], group_labels=labels, rng=0)


def test_partition_unordered_inputs() -> None:
    with pytest.raises(TypeError, match=r"items.*order"):
        random_partition({0, 1}, ratios=[1, 1], rng=0)
    with pytest.raises(TypeError, match=r"group_labels.*order"):
        random_partition([0, 1], ratios=[1, 1], group_labels={"a", "b"}, rng=0)


def test_partition_unhashable_group_labels() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        random_partition([0, 1], ratios=[1, 1], group_labels=[[0], [1]], rng=0)


@pytest.mark.parametrize("parameter", ["sizes", "ratios"])
def test_partition_empty_group_labels(parameter: str) -> None:
    spec = [0, 0] if parameter == "sizes" else [1, 1]
    assert random_partition([], **{parameter: spec}, group_labels=[], rng=0) == ([], [])
    assert random_partitions(
        [], num=2, **{parameter: spec}, group_labels=[], rng=0
    ) == [([], []), ([], [])]


@pytest.mark.parametrize("sizes", [[-1, 4], [1, 1], [5, 0]])
def test_random_partitions_invalid_sizes(sizes) -> None:
    r"""Repeated draws retain validation of non-negative, exact group counts."""
    with pytest.raises(ValueError, match="sizes"):
        random_partitions(
            range(4), num=2, sizes=sizes, group_labels=[0, 1, 0, 2], rng=0
        )
