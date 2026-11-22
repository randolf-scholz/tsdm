r"""Collection-level partitioning, reproducibility, and nested fold isolation."""

from random import getstate

import pytest
import torch

from tsdm.tasks.base import SplitType
from tsdm.tasks.utils import random_partition, repeated_random_folds, rotating_folds


def test_rotating_fold_assignments() -> None:
    r"""Pin seeded assignments and verify coverage across an entire rotation."""
    folds = rotating_folds(
        range(8),
        chunks={"train": 2, "valid": 1, "test": 1},
        num_folds=4,
        rng=2022,
    )
    assert folds[0, SplitType.TRAIN] == [0, 1, 5, 6]
    assert folds[0, SplitType.VALIDATION] == [2, 7]
    assert folds[0, SplitType.TEST] == [3, 4]
    for fold in range(4):
        parts = [
            folds[fold, split]
            for split in (SplitType.TRAIN, SplitType.VALIDATION, SplitType.TEST)
        ]
        flattened = [key for part in parts for key in part]
        assert sorted(flattened) == list(range(8))
        assert [len(part) for part in parts] == [4, 2, 2]
    for key in range(8):
        assert sum(key in folds[fold, SplitType.TEST] for fold in range(4)) == 1
        assert sum(key in folds[fold, SplitType.VALIDATION] for fold in range(4)) == 1


@pytest.mark.parametrize("policy", ["rotating", "repeated"])
def test_group_integrity_and_reproducibility(policy: str) -> None:
    r"""Unequal replicate groups stay intact; metadata mapping order is irrelevant."""
    keys = [(1, 10), (2, 20), (1, 11), (3, 30), (1, 12), (4, 40)]
    groups = {key: (key[0], "red") for key in keys}
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
            groups=dict(reversed(list(groups.items()))),
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
            groups=dict(reversed(list(groups.items()))),
            ratios={"train": 1, "test": 1},
            num_folds=2,
            rng=12,
        )
    assert folds == again
    assert getstate() == before
    assert keys == original_keys
    assert groups == original_groups
    for fold in range(2):
        train, test = folds[fold, SplitType.TRAIN], folds[fold, SplitType.TEST]
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(keys)
        assert {groups[key] for key in train}.isdisjoint(groups[key] for key in test)
        assert len({groups[key] for key in train}) == 2
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
    assert folds[0, SplitType.TRAIN] == [0, 1, 5, 6]
    assert folds[1, SplitType.TRAIN] == [1, 2, 4, 7]
    assert folds[1, SplitType.VALIDATION] == [0, 3]
    assert folds[1, SplitType.TEST] == [5, 6]


def test_nested_folds_exclude_outer_test_elements() -> None:
    outer = rotating_folds(
        range(12), chunks={"train": 2, "test": 1}, num_folds=3, rng=7
    )
    for fold in range(3):
        inner = rotating_folds(
            outer[fold, SplitType.TRAIN],
            chunks={"train": 1, "valid": 1},
            num_folds=2,
            rng=8,
            fold_prefix=(fold,),
        )
        for (outer_id, inner_id, split), members in inner.items():
            assert outer_id == fold
            assert inner_id in (0, 1)
            assert isinstance(split, SplitType)
            assert set(members) <= set(outer[fold, SplitType.TRAIN])
            assert set(members).isdisjoint(outer[fold, SplitType.TEST])


def test_largest_remainder_and_empty_splits() -> None:
    folds = repeated_random_folds(
        range(5),
        ratios={"train": 1, "valid": 1, "test": 1, "inference": 0},
        num_folds=2,
        rng=0,
        fold_prefix=(4, 2),
    )
    for fold in range(2):
        assert [
            len(folds[4, 2, fold, split])
            for split in (
                SplitType.TRAIN,
                SplitType.VALIDATION,
                SplitType.TEST,
                SplitType.INFERENCE,
            )
        ] == [2, 2, 1, 0]
    assert repeated_random_folds(
        [], ratios={"train": 1, "test": 0}, num_folds=1, rng=0
    ) == {(0, SplitType.TRAIN): [], (0, SplitType.TEST): []}


def test_empty_rotating_folds_and_boolean_identifiers() -> None:
    empty = rotating_folds([], chunks={"train": 1, "test": 0}, num_folds=1, rng=0)
    assert empty == {(0, SplitType.TRAIN): [], (0, SplitType.TEST): []}
    assert rotating_folds(
        [True, False], chunks={"train": 1, "test": 1}, num_folds=2, rng=0
    )[0, SplitType.TRAIN] == [True]


def test_extreme_ratios_are_normalized_without_overflow() -> None:
    folds = repeated_random_folds(
        range(5), ratios={"train": 1e308, "test": 1e308}, num_folds=1, rng=0
    )
    assert len(folds[0, SplitType.TRAIN]) == 3
    assert len(folds[0, SplitType.TEST]) == 2


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
def test_invalid_chunk_counts(chunks, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rotating_folds([], chunks=chunks, num_folds=2, rng=0)


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
def test_invalid_ratios(ratios, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        repeated_random_folds([], ratios=ratios, num_folds=1, rng=0)


@pytest.mark.parametrize("configuration", [{"train": 1, "training": 1}, {"unknown": 1}])
def test_invalid_split_names(configuration) -> None:
    with pytest.raises(ValueError, match="Split"):
        rotating_folds([], chunks=configuration, num_folds=1, rng=0)
    with pytest.raises(ValueError, match="Split"):
        repeated_random_folds([], ratios=configuration, num_folds=1, rng=0)


def test_duplicate_identifiers_are_rejected() -> None:
    with pytest.raises(ValueError, match="identifiers"):
        rotating_folds([1, 1], chunks={"train": 1}, num_folds=1, rng=0)
    with pytest.raises(ValueError, match="identifiers"):
        repeated_random_folds([1, 1], ratios={"train": 1}, num_folds=1, rng=0)


def test_unordered_identifiers_are_rejected() -> None:
    with pytest.raises(TypeError, match="order"):
        rotating_folds({1, 2}, chunks={"train": 1}, num_folds=1, rng=0)
    with pytest.raises(TypeError, match="order"):
        repeated_random_folds({1, 2}, ratios={"train": 1}, num_folds=1, rng=0)


@pytest.mark.parametrize("groups", [{}, {1: "a", 2: "b"}])
def test_invalid_group_mappings(groups) -> None:
    with pytest.raises(ValueError, match="groups"):
        rotating_folds([1], groups=groups, chunks={"train": 1}, num_folds=1, rng=0)


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
