from collections.abc import KeysView
from typing import Any

import pytest
from pandas import Series

from tsdm.tasks.base import SplitType, TimeSeriesTask

type SplitKey = tuple[int, str]


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("train", SplitType.TRAIN),
        ("training", SplitType.TRAIN),
        ("trainval", SplitType.TRAIN_VALIDATION),
        ("train_val", SplitType.TRAIN_VALIDATION),
        ("train_valid", SplitType.TRAIN_VALIDATION),
        ("train_validation", SplitType.TRAIN_VALIDATION),
        ("valid", SplitType.VALIDATION),
        ("validation", SplitType.VALIDATION),
        ("val", SplitType.VALIDATION),
        ("test", SplitType.TEST),
        ("testing", SplitType.TEST),
        ("infer", SplitType.INFERENCE),
        ("inference", SplitType.INFERENCE),
    ],
)
def test_split_type_aliases(alias: str, expected: SplitType) -> None:
    r"""Test that SplitType supports all lowercase and uppercase aliases."""
    assert SplitType(alias) is expected
    assert SplitType(alias.upper()) is expected


@pytest.fixture
def task() -> TimeSeriesTask[Any]:
    r"""Create an uninitialized task with all standard split types."""

    class DummyTask(TimeSeriesTask[Any]):
        pass

    folds = {
        "train": Series(dtype=bool),
        "trainval": Series(dtype=bool),
        "valid": Series(dtype=bool),
        "test": Series(dtype=bool),
    }
    return DummyTask(
        NotImplemented,
        folds=folds,
        initialize=False,
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("training", SplitType.TRAIN),
        ("train_valid", SplitType.TRAIN_VALIDATION),
        ("val", SplitType.VALIDATION),
        ("testing", SplitType.TEST),
        ("infer", SplitType.INFERENCE),
        ((0, "training"), SplitType.TRAIN),
        ((0, "train_val"), SplitType.TRAIN_VALIDATION),
        ((0, "val"), SplitType.VALIDATION),
        ((3, "train", "test"), SplitType.TEST),
        ([3, "testing"], SplitType.TEST),
    ],
)
def test_time_series_task_split_type(
    task: TimeSeriesTask[Any], key: object, expected: SplitType
) -> None:
    r"""Test that TimeSeriesTask classifies split keys using SplitType."""
    assert task.split_type(key) is expected
    assert task.is_train_split(key) is (
        expected in {SplitType.TRAIN, SplitType.TRAIN_VALIDATION}
    )


@pytest.mark.parametrize("key", ["other", (0, "other")])
def test_time_series_task_rejects_unknown_split(
    task: TimeSeriesTask[Any], key: object
) -> None:
    r"""Test that unknown split names raise an error."""
    with pytest.raises(ValueError, match="valid SplitType"):
        task.split_type(key)


def test_train_split_mapping(task: TimeSeriesTask[Any]) -> None:
    r"""Test that trainval uses itself as its associated training split."""
    assert {key: task.get_train_split(key) for key in tuple(task.folds.keys())} == {
        "train": "train",
        "trainval": "trainval",
        "valid": "train",
        "test": "train",
    }


def test_train_split_mapping_uses_only_fold_keys() -> None:
    r"""Test tuple split keys without relying on the folds' implementation type."""

    class FoldTable:
        def __init__(self, data: dict[SplitKey, Series], /) -> None:
            self.data = data
            self.getitem_calls = 0

        def keys(self) -> KeysView[SplitKey]:
            return self.data.keys()

        def __getitem__(self, key: SplitKey, /) -> Series:
            self.getitem_calls += 1
            return self.data[key]

    class DummyTask(TimeSeriesTask[SplitKey]):
        pass

    folds = FoldTable(
        {
            (0, "train"): Series(dtype=bool),
            (0, "valid"): Series(dtype=bool),
            (0, "test"): Series(dtype=bool),
            (1, "train"): Series(dtype=bool),
            (1, "trainval"): Series(dtype=bool),
            (1, "test"): Series(dtype=bool),
        }
    )
    task = DummyTask(
        NotImplemented,
        folds=folds,
        index=list(folds.keys()),
        initialize=False,
    )

    assert {key: task.get_train_split(key) for key in tuple(folds.keys())} == {
        (0, "train"): (0, "train"),
        (0, "valid"): (0, "train"),
        (0, "test"): (0, "train"),
        (1, "train"): (1, "train"),
        (1, "trainval"): (1, "trainval"),
        (1, "test"): (1, "train"),
    }
    assert folds.getitem_calls == 0
