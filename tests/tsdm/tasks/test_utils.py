r"""Collection-level partitioning, reproducibility, and nested fold isolation."""

from collections.abc import Callable
from functools import partial
from random import getstate
from types import MappingProxyType

import pytest
import torch

from tsdm.tasks.base import SplitType
from tsdm.tasks.utils import (
    random_partition,
    repeated_random_folds,
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
                rotating_folds, chunks={"train": 1, "test": 1}, num_folds=2, rng=12
            )
        case "repeated":
            return partial(
                repeated_random_folds,
                ratios={"train": 1, "test": 1},
                num_folds=2,
                rng=12,
            )
        case _:
            raise AssertionError(request.param)


def test_rotating_fold_assignments() -> None:
    r"""Pin seeded assignments and verify coverage across an entire rotation."""
    folds = rotating_folds(
        range(8),
        chunks={"train": 2, "valid": 1, "test": 1},
        num_folds=4,
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


@pytest.mark.parametrize("policy", ["rotating", "repeated"])
def test_group_integrity_and_reproducibility(policy: str) -> None:
    r"""Unequal replicate groups stay intact with positionally aligned labels."""
    keys = [(1, 10), (2, 20), (1, 11), (3, 30), (1, 12), (4, 40)]
    groups = [(key[0], "red") for key in keys]
    group_by_key = dict(zip(keys, groups, strict=True))
    original_keys, original_groups = keys.copy(), groups.copy()
    before = getstate()
    if policy == "rotating":
        folds = rotating_folds(
            keys,
            groups=groups,
            chunks={"train": 1, "test": 1},
            num_folds=2,
            rng=12,
        )
        again = rotating_folds(
            tuple(keys),
            groups=tuple(groups),
            chunks={"train": 1, "test": 1},
            num_folds=2,
            rng=12,
        )
    else:
        folds = repeated_random_folds(
            keys,
            groups=groups,
            ratios={"train": 1, "test": 1},
            num_folds=2,
            rng=12,
        )
        again = repeated_random_folds(
            tuple(keys),
            groups=tuple(groups),
            ratios={"train": 1, "test": 1},
            num_folds=2,
            rng=12,
        )
    assert folds == again
    assert getstate() == before
    assert keys == original_keys
    assert groups == original_groups
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
    folds = repeated_random_folds(
        range(8),
        ratios={"train": 2, "valid": 1, "test": 1},
        num_folds=2,
        rng=2022,
    )
    assert folds[0]["train"] == [0, 1, 5, 6]
    assert folds[1]["train"] == [1, 2, 4, 7]
    assert folds[1]["valid"] == [0, 3]
    assert folds[1]["test"] == [5, 6]


def test_nested_folds_exclude_outer_test_elements() -> None:
    outer = rotating_folds(
        range(12), chunks={"train": 2, "test": 1}, num_folds=3, rng=7
    )
    for fold in range(3):
        inner = rotating_folds(
            outer[fold]["train"],
            chunks={"train": 1, "valid": 1},
            num_folds=2,
            rng=8,
        )
        assert len(inner) == 2
        for partition in inner:
            for members in partition.values():
                assert set(members) <= set(outer[fold]["train"])
                assert set(members).isdisjoint(outer[fold]["test"])


def test_largest_remainder_and_empty_splits() -> None:
    folds = repeated_random_folds(
        range(5),
        ratios={"train": 1, "valid": 1, "test": 1, "inference": 0},
        num_folds=2,
        rng=0,
    )
    for fold in range(2):
        assert [
            len(folds[fold][split]) for split in ("train", "valid", "test", "inference")
        ] == [2, 2, 1, 0]
    assert repeated_random_folds(
        [], ratios={"train": 1, "test": 0}, num_folds=1, rng=0
    ) == [{"train": [], "test": []}]


def test_empty_rotating_folds_and_boolean_identifiers() -> None:
    empty = rotating_folds([], chunks={"train": 1, "test": 0}, num_folds=1, rng=0)
    assert empty == [{"train": [], "test": []}]
    assert rotating_folds(
        [True, False], chunks={"train": 1, "test": 1}, num_folds=2, rng=0
    )[0]["train"] == [True]


def test_extreme_ratios_are_normalized_without_overflow() -> None:
    folds = repeated_random_folds(
        range(5), ratios={"train": 1e308, "test": 1e308}, num_folds=1, rng=0
    )
    assert len(folds[0]["train"]) == 3
    assert len(folds[0]["test"]) == 2


@pytest.mark.parametrize("rng", [None, 1.5, "42"])
def test_rng_must_be_an_integer_or_random(rng) -> None:
    with pytest.raises(TypeError, match="rng"):
        rotating_folds([1], chunks={"train": 1}, num_folds=1, rng=rng)
    with pytest.raises(TypeError, match="rng"):
        repeated_random_folds([1], ratios={"train": 1}, num_folds=1, rng=rng)


def test_rng_is_required() -> None:
    with pytest.raises(TypeError, match="rng"):
        rotating_folds([1], chunks={"train": 1}, num_folds=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="rng"):
        repeated_random_folds([1], ratios={"train": 1}, num_folds=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("num_folds", [0, -1, 1.5])
def test_invalid_fold_counts(num_folds) -> None:
    with pytest.raises(ValueError, match="num_folds"):
        rotating_folds([], chunks={"train": 2}, num_folds=num_folds, rng=0)
    with pytest.raises(ValueError, match="num_folds"):
        repeated_random_folds([], ratios={"train": 1}, num_folds=num_folds, rng=0)


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ({}, "At least one split must be configured"),
        ({"train": 0}, "positive weight"),
        ({"train": -1}, "non-negative"),
        ({"train": 1.5}, "integer counts"),
        ({"train": 3}, "divisible by num_folds"),
    ],
)
@pytest.mark.parametrize("named", [True, False])
def test_invalid_chunk_counts(chunks, message: str, *, named: bool) -> None:
    with pytest.raises(ValueError, match=message):
        rotating_folds(
            [], chunks=chunks if named else list(chunks.values()), num_folds=2, rng=0
        )


@pytest.mark.parametrize(
    ("ratios", "message"),
    [
        ({}, "At least one split must be configured"),
        ({"train": 0}, "positive weight"),
        ({"train": -1}, "non-negative"),
        ({"train": float("nan")}, "finite"),
        ({"train": float("inf")}, "finite"),
    ],
)
@pytest.mark.parametrize("named", [True, False])
def test_invalid_ratios(ratios, message: str, *, named: bool) -> None:
    with pytest.raises(ValueError, match=message):
        repeated_random_folds(
            [], ratios=ratios if named else list(ratios.values()), num_folds=1, rng=0
        )


@pytest.mark.parametrize("configuration", [{"train": 1, "training": 1}, {"unknown": 1}])
def test_arbitrary_split_names(configuration) -> None:
    r"""Former aliases and unknown names are independent, unchanged split keys."""
    expected = [{key: [] for key in configuration}]
    assert rotating_folds([], chunks=configuration, num_folds=1, rng=0) == expected
    assert (
        repeated_random_folds([], ratios=configuration, num_folds=1, rng=0) == expected
    )


@pytest.mark.parametrize(
    ("fold_function", "parameter"),
    [(rotating_folds, "chunks"), (repeated_random_folds, "ratios")],
)
@pytest.mark.parametrize("groups", [None, [0, 0, 1, 1, 2, 2, 3, 3]])
def test_named_and_unnamed_folds(fold_function, parameter: str, *, groups) -> None:
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
        items, **{parameter: spec}, num_folds=2, rng=12, groups=groups
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
            items, **{parameter: counts}, num_folds=2, rng=12, groups=groups
        )
        assert isinstance(unnamed, list)
        assert unnamed == [tuple(partition.values()) for partition in named]
        assert all(isinstance(partition, tuple) for partition in unnamed)
        for partition in unnamed:
            assert len(partition) == len(counts)


@pytest.mark.parametrize(
    ("fold_function", "parameter"),
    [(rotating_folds, "chunks"), (repeated_random_folds, "ratios")],
)
def test_unnamed_empty_folds(fold_function, parameter: str) -> None:
    r"""Empty inputs retain every fold and split, with independently mutable lists."""
    folds = fold_function([], **{parameter: [2, 0]}, num_folds=2, rng=0)
    assert folds == [([], []), ([], [])]
    folds[0][0].append(1)
    assert folds == [([1], []), ([], [])]


@pytest.mark.parametrize("groups", [None, [0, 1, 0, 2, 0, 3, 2, 3]])
def test_unhashable_occurrences_match_index_partitions(
    fold_generator: Callable[..., list[dict[str, list]]], groups
) -> None:
    r"""Equal unhashable items retain identity, multiplicity, order, and allocation."""
    items = [[index % 2] for index in range(8)]
    items[-1] = items[0]  # Repeated references are separate occurrences too.
    original = items.copy()
    folds = fold_generator(items, groups=groups)
    expected = fold_generator(range(len(items)), groups=groups)
    assert folds == fold_generator(items, groups=groups)
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
    folds = fold_generator(items, groups=items)
    for fold in range(2):
        train, test = folds[fold]["train"], folds[fold]["test"]
        assert sorted(train + test) == sorted(items)
        assert set(train).isdisjoint(test)
        assert sorted([train, test]) == [[1, 1, 1], [2, 2, 2]]


def test_unordered_identifiers_are_rejected() -> None:
    with pytest.raises(TypeError, match="order"):
        rotating_folds({1, 2}, chunks={"train": 1}, num_folds=1, rng=0)
    with pytest.raises(TypeError, match="order"):
        repeated_random_folds({1, 2}, ratios={"train": 1}, num_folds=1, rng=0)


@pytest.mark.parametrize("groups", [[], ["a", "b"]])
def test_invalid_group_lengths(
    fold_generator: Callable[..., list[dict[str, list]]], groups
) -> None:
    with pytest.raises(ValueError, match="groups"):
        fold_generator([1], groups=groups)


def test_unordered_labels_are_rejected(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    with pytest.raises(TypeError, match=r"groups.*order"):
        fold_generator([1, 2], groups={"a", "b"})


def test_unhashable_labels_are_rejected(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    with pytest.raises(TypeError, match="unhashable"):
        fold_generator([1, 2], groups=[[1], [2]])


def test_empty_group_labels(
    fold_generator: Callable[..., list[dict[str, list]]],
) -> None:
    assert fold_generator([], groups=[]) == fold_generator([])


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
