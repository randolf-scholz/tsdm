import pytest
from pandas import Series

from tsdm.tasks.base import SplitType, TimeSeriesTask


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
def task() -> TimeSeriesTask[object]:
    r"""Create an uninitialized task with all standard split types."""

    class DummyTask(TimeSeriesTask[object]):
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
        ("other", SplitType.UNKNOWN),
        ((0, "training"), SplitType.TRAIN),
        ((0, "train_val"), SplitType.TRAIN_VALIDATION),
        ((0, "val"), SplitType.VALIDATION),
    ],
)
def test_time_series_task_split_type(
    task: TimeSeriesTask[object], key: object, expected: SplitType
) -> None:
    r"""Test that TimeSeriesTask classifies split keys using SplitType."""
    assert task.split_type(key) is expected
    if expected is SplitType.UNKNOWN:
        with pytest.raises(ValueError, match="Unknown split type"):
            task.is_train_split(key)
        return

    assert task.is_train_split(key) is (
        expected in {SplitType.TRAIN, SplitType.TRAIN_VALIDATION}
    )


def test_time_series_task_rejects_mixed_split_types(
    task: TimeSeriesTask[object],
) -> None:
    r"""Test that a composite key cannot mix training and inference splits."""
    with pytest.raises(ValueError, match="both training and inference"):
        task.split_type(("train", "test"))


def test_train_split_mapping(task: TimeSeriesTask[object]) -> None:
    r"""Test that trainval uses itself as its associated training split."""
    assert task.train_split == {
        "train": "train",
        "trainval": "trainval",
        "valid": "train",
        "test": "train",
    }
