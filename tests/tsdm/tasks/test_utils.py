r"""Collection-level partitioning, reproducibility, and nested fold isolation."""

from collections.abc import Callable
from functools import partial
from random import Random, getstate
from types import MappingProxyType

import pytest
import torch

from tsdm.tasks.base import SplitType
from tsdm.tasks.utils import (
    random_partition,
    random_partitions,
    rotating_folds,
)


@pytest.fixture(params=["rotating", "repeated"])
def fold_generator(
    request: pytest.FixtureRequest,
) -> Callable[..., list[dict[str, list]]]:
    r"""Exercise occurrence and label semantics with both fold algorithms."""
    match request.param:
        case "rotating":
            return partial(
                rotating_folds,
                chunks={"train": 1, "test": 1},
                num=2,
                rng=12,
            )
        case "repeated":
            return partial(
                random_partitions,
                ratios={"train": 1, "test": 1},
                num=2,
                rng=12,
            )
        case _:
            raise AssertionError(request.param)


def test_rotating_fold_assignments() -> None:
    r"""Pin seeded assignments and verify coverage across an entire rotation."""
    folds = rotating_folds(
        range(8),
        chunks={"train": 2, "valid": 1, "test": 1},
        num=4,
        rng=2022,
    )
    assert folds[0]["train"] == [0, 1, 5, 6]
    assert folds[0]["valid"] == [2, 7]
    assert folds[0]["test"] == [3, 4]
    for fold in range(4):
        parts = [folds[fold][split] for split in ("train", "valid", "test")]
        flattened = [key for part in parts for key in part]
        assert sorted(flattened) == list(range(8))
        assert [len(part) for part in parts] == [4, 2, 2]
    for key in range(8):
        assert sum(key in folds[fold]["test"] for fold in range(4)) == 1
        assert sum(key in folds[fold]["valid"] for fold in range(4)) == 1


@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
def test_rotating_single_partition(*, named: bool, grouped: bool) -> None:
    r"""The default returns a one-element list that supports partition unpacking."""
    marker = object()
    chunks = (
        MappingProxyType({marker: 1, "test": 1, "empty": 0}) if named else [1, 1, 0]
    )
    items = [[index] for index in range(8)]
    labels = [0, 1, 0, 2, 0, 3, 2, 3] if grouped else None
    rng = Random(12)
    reference_rng = Random(12)
    default = rotating_folds(items, chunks=chunks, group_labels=labels, rng=rng)
    singleton = rotating_folds(
        items, num=1, chunks=chunks, group_labels=labels, rng=reference_rng
    )
    multiple = rotating_folds(items, num=2, chunks=chunks, group_labels=labels, rng=12)
    assert isinstance(default, list)
    assert isinstance(singleton, list)
    assert default == singleton
    [single] = default
    assert isinstance(single, dict if named else tuple)
    assert single == multiple[0]
    assert rng.getstate() == reference_rng.getstate()
    if isinstance(single, dict):
        assert list(single) == list(chunks)
        assert next(iter(single)) is marker
        assert single["empty"] == []
    else:
        assert single[-1] == []


@pytest.mark.parametrize("named", [True, False])
def test_rotating_single_empty_partition(*, named: bool) -> None:
    chunks = {"train": 1, "test": 1} if named else [1, 1]
    expected = {"train": [], "test": []} if named else ([], [])
    assert rotating_folds([], chunks=chunks, rng=0) == [expected]


def test_rotating_single_partition_error_check() -> None:
    r"""The default retains sample-proportion checks for unequal groups."""
    with pytest.raises(RuntimeError, match="total variation distance"):
        rotating_folds(
            range(10),
            chunks=[1, 1],
            group_labels=[0] * 9 + [1],
            rng=0,
            max_error=0.05,
        )


@pytest.mark.parametrize("policy", ["rotating", "repeated"])
def test_group_integrity_and_reproducibility(policy: str) -> None:
    r"""Unequal replicate groups stay intact with positionally aligned labels."""
    keys = [(1, 10), (2, 20), (1, 11), (3, 30), (1, 12), (4, 40)]
    group_labels = [(key[0], "red") for key in keys]
    group_by_key = dict(zip(keys, group_labels, strict=True))
    original_keys, original_groups = keys.copy(), group_labels.copy()
    before = getstate()
    if policy == "rotating":
        folds = rotating_folds(
            keys,
            group_labels=group_labels,
            chunks={"train": 1, "test": 1},
            num=2,
            rng=12,
        )
        again = rotating_folds(
            tuple(keys),
            group_labels=tuple(group_labels),
            chunks={"train": 1, "test": 1},
            num=2,
            rng=12,
        )
    else:
        folds = random_partitions(
            keys,
            group_labels=group_labels,
            ratios={"train": 1, "test": 1},
            num=2,
            rng=12,
        )
        again = random_partitions(
            tuple(keys),
            group_labels=tuple(group_labels),
            ratios={"train": 1, "test": 1},
            num=2,
            rng=12,
        )
    assert folds == again
    assert getstate() == before
    assert keys == original_keys
    assert group_labels == original_groups
    for fold in range(2):
        train, test = folds[fold]["train"], folds[fold]["test"]
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(keys)
        assert {group_by_key[key] for key in train}.isdisjoint(
            group_by_key[key] for key in test
        )
        assert len({group_by_key[key] for key in train}) == 2
        assert train == [key for key in keys if key in train]
        assert test == [key for key in keys if key in test]


def test_repeated_folds_advance_rng() -> None:
    r"""A fixed seed reproduces the full sequence without resetting each fold."""
    folds = random_partitions(
        range(8),
        ratios={"train": 2, "valid": 1, "test": 1},
        num=2,
        rng=2022,
    )
    assert folds[0]["train"] == [0, 1, 5, 6]
    assert folds[1]["train"] == [1, 2, 4, 7]
    assert folds[1]["valid"] == [0, 3]
    assert folds[1]["test"] == [5, 6]


def test_nested_folds_exclude_outer_test_elements() -> None:
    outer = rotating_folds(range(12), chunks={"train": 2, "test": 1}, num=3, rng=7)
    for fold in range(3):
        inner = rotating_folds(
            outer[fold]["train"],
            chunks={"train": 1, "valid": 1},
            num=2,
            rng=8,
        )
        assert len(inner) == 2
        for partition in inner:
            for members in partition.values():
                assert set(members) <= set(outer[fold]["train"])
                assert set(members).isdisjoint(outer[fold]["test"])


def test_largest_remainder_and_empty_splits() -> None:
    folds = random_partitions(
        range(5),
        ratios={"train": 1, "valid": 1, "test": 1, "inference": 0},
        num=2,
        rng=0,
    )
    for fold in range(2):
        assert [
            len(folds[fold][split]) for split in ("train", "valid", "test", "inference")
        ] == [2, 2, 1, 0]


def test_rotating_folds_with_boolean_items() -> None:
    assert rotating_folds([True, False], chunks={"train": 1, "test": 1}, num=2, rng=0)[
        0
    ]["train"] == [True]


def test_extreme_ratios_are_normalized_without_overflow() -> None:
    folds = random_partitions(
        range(5),
        ratios={"train": 1e308, "test": 1e308},
        num=1,
        rng=0,
    )
    assert len(folds[0]["train"]) == 3
    assert len(folds[0]["test"]) == 2


@pytest.mark.parametrize("rng", [None, 1.5, "42"])
def test_rng_must_be_an_integer_or_random(rng) -> None:
    with pytest.raises(TypeError, match="rng"):
        rotating_folds([1], chunks={"train": 1}, num=1, rng=rng)
    with pytest.raises(TypeError, match="rng"):
        random_partitions([1], ratios={"train": 1}, num=1, rng=rng)


def test_rng_is_required() -> None:
    with pytest.raises(TypeError, match="rng"):
        rotating_folds([1], chunks={"train": 1}, num=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="rng"):
        random_partitions([1], ratios={"train": 1}, num=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("num", [None, 0, -1, 1.5])
def test_invalid_fold_counts(num) -> None:
    with pytest.raises(ValueError, match="num"):
        rotating_folds([], chunks={"train": 2}, num=num, rng=0)
    with pytest.raises(ValueError, match="num"):
        random_partitions([], ratios={"train": 1}, num=num, rng=0)


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ({}, "At least one split must be configured"),
        ({"train": 0}, "positive weight"),
        ({"train": -1}, "non-negative"),
        ({"train": 1.5}, "integer counts"),
        ({"train": 3}, "divisible by num"),
    ],
)
@pytest.mark.parametrize("named", [True, False])
def test_invalid_chunk_counts(chunks, message: str, *, named: bool) -> None:
    with pytest.raises(ValueError, match=message):
        rotating_folds(
            [], chunks=chunks if named else list(chunks.values()), num=2, rng=0
        )


@pytest.mark.parametrize(
    ("ratios", "message"),
    [
        ({}, "ratios must not all be zero"),
        ({"train": 0}, "ratios must not all be zero"),
        ({"train": -1}, "non-negative"),
        ({"train": float("nan")}, "finite"),
        ({"train": float("inf")}, "finite"),
    ],
)
@pytest.mark.parametrize("named", [True, False])
def test_invalid_ratios(ratios, message: str, *, named: bool) -> None:
    with pytest.raises(ValueError, match=message):
        random_partitions(
            [], ratios=ratios if named else list(ratios.values()), num=1, rng=0
        )


@pytest.mark.parametrize("configuration", [{"train": 1, "training": 1}, {"unknown": 1}])
def test_arbitrary_split_names(configuration) -> None:
    r"""Former aliases and unknown names are independent, unchanged split keys."""
    items = range(sum(configuration.values()))
    rotating = rotating_folds(items, chunks=configuration, num=1, rng=0)
    repeated = random_partitions(items, ratios=configuration, num=1, rng=0)
    assert list(rotating[0]) == list(configuration)
    assert list(repeated[0]) == list(configuration)


@pytest.mark.parametrize(
    ("fold_function", "parameter"),
    [(rotating_folds, "chunks"), (random_partitions, "ratios")],
)
@pytest.mark.parametrize("group_labels", [None, [0, 0, 1, 1, 2, 2, 3, 3]])
def test_named_and_unnamed_folds(
    fold_function, parameter: str, *, group_labels
) -> None:
    r"""Both forms have identical assignments, with exact keys and empty splits."""
    marker = object()
    spec = MappingProxyType(
        {
            "train": 1,
            "training": 1,
            SplitType.VALIDATION: 1,
            ("custom", 1): 1,
            marker: 0,
            17: 0,
            None: 0,
        }
    )
    items = [[index] for index in range(8)]
    named = fold_function(
        items, **{parameter: spec}, num=2, rng=12, group_labels=group_labels
    )
    assert isinstance(named, list)
    assert len(named) == 2
    for partition in named:
        assert isinstance(partition, dict)
        assert all(actual is key for actual, key in zip(partition, spec, strict=True))
        assert partition[marker] == partition[17] == partition[None] == []
        assert sorted(item[0] for part in partition.values() for item in part) == list(
            range(8)
        )
    for counts in (list(spec.values()), tuple(spec.values())):
        unnamed = fold_function(
            items, **{parameter: counts}, num=2, rng=12, group_labels=group_labels
        )
        assert isinstance(unnamed, list)
        assert unnamed == [tuple(partition.values()) for partition in named]
        assert all(isinstance(partition, tuple) for partition in unnamed)
        for partition in unnamed:
            assert len(partition) == len(counts)


@pytest.mark.parametrize(
    ("fold_function", "parameter"),
    [(rotating_folds, "chunks"), (random_partitions, "ratios")],
)
def test_unnamed_zero_weight_splits(fold_function, parameter: str) -> None:
    r"""Zero-weight splits remain empty, with independently mutable item lists."""
    folds = fold_function([1, 2], **{parameter: [2, 0]}, num=2, rng=0)
    assert folds == [([1, 2], []), ([1, 2], [])]
    folds[0][0].append(3)
    assert folds == [([1, 2, 3], []), ([1, 2], [])]


@pytest.mark.parametrize(
    ("fold_function", "parameter"),
    [(rotating_folds, "chunks"), (random_partitions, "ratios")],
)
@pytest.mark.parametrize("named", [True, False])
def test_group_error_measures_items(
    fold_function, parameter: str, *, named: bool
) -> None:
    r"""Balanced group counts do not conceal highly imbalanced item counts."""
    spec = {"train": 1, "test": 1} if named else [1, 1]
    partitioner = partial(
        fold_function,
        range(10),
        **{parameter: spec},
        group_labels=[0] * 9 + [1],
        num=2,
        rng=0,
    )
    assert partitioner() == partitioner(max_error=None)
    with pytest.raises(RuntimeError, match=r"total variation distance 0.4"):
        partitioner(max_error=0.05)


@pytest.mark.parametrize(
    ("fold_function", "parameter"),
    [(rotating_folds, "chunks"), (random_partitions, "ratios")],
)
def test_group_error_aggregates_all_splits(fold_function, parameter: str) -> None:
    r"""Four splits have TV 0.25 even though each proportion differs by only 0.125."""
    with pytest.raises(RuntimeError, match=r"total variation distance 0.25"):
        fold_function(
            range(8),
            **{parameter: [1, 1, 1, 1]},
            group_labels=[0, 0, 0, 1, 1, 1, 2, 3],
            num=2,
            rng=0,
            max_error=0.1875,
        )


def test_rotating_error_checks_later_folds() -> None:
    r"""Fold zero has TV 0.05, but fold one has TV 0.15 and must reject the batch."""
    with pytest.raises(RuntimeError, match=r"total variation distance 0.15"):
        rotating_folds(range(10), chunks=[6, 2], num=4, rng=0, max_error=0.125)


def test_repeated_error_checks_later_folds() -> None:
    r"""Fold zero balances items exactly; fold one has TV 0.125 despite equal groups."""
    with pytest.raises(RuntimeError, match=r"total variation distance 0.125"):
        random_partitions(
            range(8),
            ratios=[1, 1],
            group_labels=[0, 0, 0, 1, 2, 2, 3, 3],
            num=2,
            rng=6,
            max_error=0.05,
        )


@pytest.mark.parametrize("group_labels", [None, [0, 1, 0, 2, 0, 3, 2, 3]])
def test_unhashable_occurrences_match_index_partitions(
    fold_generator: Callable[..., list[dict[str, list]]], group_labels
) -> None:
    r"""Equal unhashable items retain identity, multiplicity, order, and allocation."""
    items = [[index % 2] for index in range(8)]
    items[-1] = items[0]  # Repeated references are separate occurrences too.
    original = items.copy()
    folds = fold_generator(items, group_labels=group_labels)
    expected = fold_generator(range(len(items)), group_labels=group_labels)
    assert folds == fold_generator(items, group_labels=group_labels)
    assert items == original
    for actual_fold, expected_fold in zip(folds, expected, strict=True):
        for key, indices in expected_fold.items():
            actual = actual_fold[key]
            assert len(actual) == len(indices)
            assert all(
                item is items[index]
                for item, index in zip(actual, indices, strict=True)
            )


def test_duplicate_items_split_independently(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    r"""Equal items split independently unless equal labels explicitly group them."""
    items = [1, 1, 1, 1]
    folds = fold_generator(items)
    assert all(part == [1, 1] for fold in folds for part in fold.values())


def test_equal_items_can_be_used_as_labels(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    r"""Passing items as labels keeps equal values together without losing copies."""
    items = [1, 2, 1, 2, 1, 2]
    folds = fold_generator(items, group_labels=items)
    for fold in range(2):
        train, test = folds[fold]["train"], folds[fold]["test"]
        assert sorted(train + test) == sorted(items)
        assert set(train).isdisjoint(test)
        assert sorted([train, test]) == [[1, 1, 1], [2, 2, 2]]


def test_unordered_identifiers_are_rejected() -> None:
    with pytest.raises(TypeError, match="order"):
        rotating_folds({1, 2}, chunks={"train": 1}, num=1, rng=0)
    with pytest.raises(TypeError, match="order"):
        random_partitions({1, 2}, ratios={"train": 1}, num=1, rng=0)


@pytest.mark.parametrize("group_labels", [[], ["a", "b"]])
def test_invalid_group_lengths(
    fold_generator: Callable[..., list[dict[str, list]]], group_labels
) -> None:
    with pytest.raises(ValueError, match="group_labels"):
        fold_generator([1], group_labels=group_labels)


def test_unordered_labels_are_rejected(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    with pytest.raises(TypeError, match=r"group_labels.*order"):
        fold_generator([1, 2], group_labels={"a", "b"})


def test_unhashable_labels_are_rejected(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    with pytest.raises(TypeError, match="unhashable"):
        fold_generator([1, 2], group_labels=[[1], [2]])


def test_empty_group_labels(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    assert fold_generator([], group_labels=[]) == fold_generator([])


def test_random_partition_dictionary_keys() -> None:
    r"""Partition dictionary keys into train, validation, and test datasets."""
    dataset = {f"sample-{index}": index for index in range(10)}
    train_keys, validation_keys, test_keys = random_partition(
        dataset.keys(), sizes=[4, 4, 2], rng=0
    )

    train_set = {key: dataset[key] for key in train_keys}
    validation_set = {key: dataset[key] for key in validation_keys}
    test_set = {key: dataset[key] for key in test_keys}

    assert tuple(map(len, (train_set, validation_set, test_set))) == (4, 4, 2)
    assert train_set.keys().isdisjoint(validation_set)
    assert train_set.keys().isdisjoint(test_set)
    assert validation_set.keys().isdisjoint(test_set)
    assert train_set.keys() | validation_set.keys() | test_set.keys() == dataset.keys()


def test_random_partition_tensor_and_labels() -> None:
    r"""Partition paired tensor features and string labels by shared indices."""
    features = torch.arange(30).reshape(10, 3)
    labels = [f"label-{index}" for index in range(len(features))]
    train_indices, validation_indices, test_indices = random_partition(
        range(len(features)), ratios=[0.4, 0.4, 0.2], rng=0
    )

    train_set = (features[train_indices], [labels[index] for index in train_indices])
    validation_set = (
        features[validation_indices],
        [labels[index] for index in validation_indices],
    )
    test_set = (features[test_indices], [labels[index] for index in test_indices])

    assert tuple(
        features.shape for features, _ in (train_set, validation_set, test_set)
    ) == ((4, 3), (4, 3), (2, 3))
    assert tuple(
        len(labels) for _, labels in (train_set, validation_set, test_set)
    ) == (4, 4, 2)
    assert all(
        torch.equal(row, features[int(label.removeprefix("label-"))])
        for dataset_features, dataset_labels in (train_set, validation_set, test_set)
        for row, label in zip(dataset_features, dataset_labels, strict=True)
    )
