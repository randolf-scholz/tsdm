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
        ((0, "training"), SplitType.TRAIN),
        ((0, "train_val"), SplitType.TRAIN_VALIDATION),
        ((0, "val"), SplitType.VALIDATION),
        ((3, "train", "test"), SplitType.TEST),
        ([3, "testing"], SplitType.TEST),
    ],
)
def test_time_series_task_split_type(
    task: TimeSeriesTask[object], key: object, expected: SplitType
) -> None:
    r"""Test that TimeSeriesTask classifies split keys using SplitType."""
    assert task.split_type(key) is expected
    assert task.is_train_split(key) is (
        expected in {SplitType.TRAIN, SplitType.TRAIN_VALIDATION}
    )


@pytest.mark.parametrize("key", ["other", (0, "other")])
def test_time_series_task_rejects_unknown_split(
    task: TimeSeriesTask[object], key: object
) -> None:
    r"""Test that unknown split names raise an error."""
    with pytest.raises(ValueError, match="valid SplitType"):
        task.split_type(key)


def test_train_split_mapping(task: TimeSeriesTask[object]) -> None:
    r"""Test that trainval uses itself as its associated training split."""
    assert task.train_partition_mapper == {
        "train": "train",
        "trainval": "trainval",
        "valid": "train",
        "test": "train",
    }
