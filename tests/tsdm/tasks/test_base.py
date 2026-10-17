from collections.abc import Iterable, Sized

import pytest

from tsdm.random.samplers import RandomSampler, Sampler
from tsdm.tasks.base import SplitType, Task, TimeSeriesTask


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


class DummyTask(TimeSeriesTask[int, list[int], int, int]):
    @property
    def split_keys(self) -> tuple[int, ...]:
        return (10, 20)

    def is_train_split(self, key: int) -> bool:
        return key == 10

    def make_partition(self, key: int, /) -> list[int]:
        return self.dataset[:3] if self.is_train_split(key) else self.dataset[3:]

    def make_generator(self, key: int, /) -> list[int]:
        return self.partitions[key]

    def make_sampler(self, key: int, /) -> Sampler[int]:
        return RandomSampler(range(len(self.partitions[key])), shuffle=False)


def test_task_with_opaque_split_keys() -> None:
    r"""Subclasses can supply split semantics independently of key structure."""
    task = DummyTask([1, 2, 3, 4, 5], error_fn=lambda x, y: abs(x - y))
    assert isinstance(task, Task)
    assert task.split_keys == (10, 20)
    assert len(task.split_keys) == 2
    assert not isinstance(task, Iterable)
    assert not isinstance(task, Sized)
    assert set(task.partitions) == {10, 20}
    assert task.partitions[10] == [1, 2, 3]
    assert task.partitions[20] == [4, 5]
    assert task.error_fn(2, 5) == 3
    assert task.collate_fns[10]([1, 2]) == [1, 2]
    assert list(task.make_dataloader(10, batch_size=2, pin_memory=False)) == [[1, 2]]
    assert list(task.make_dataloader(20, batch_size=2, pin_memory=False)) == [[4, 5]]
    assert list(
        task.make_dataloader(
            10, batch_size=2, drop_last=False, collate_fn=sum, pin_memory=False
        )
    ) == [3, 3]


def test_task_rejects_empty_split_keys() -> None:
    r"""An empty key collection fails before any lazy mappings are created."""

    class EmptyTask(DummyTask):
        @property
        def split_keys(self) -> tuple[int, ...]:
            return ()

    with pytest.raises(
        RuntimeError, match=r"split_keys.*non-empty.*circular dependency"
    ):
        EmptyTask([], error_fn=lambda x, y: abs(x - y))


@pytest.mark.parametrize(
    "mapping_name",
    ["partitions", "generators", "samplers", "collate_fns", "dataloaders"],
)
def test_task_rejects_split_keys_depending_on_lazy_mappings(mapping_name: str) -> None:
    r"""Explain the circular dependency instead of exposing an AttributeError."""

    class CircularTask(DummyTask):
        @property
        def split_keys(self) -> tuple[int, ...]:
            return tuple(getattr(self, mapping_name))

    with pytest.raises(RuntimeError, match="circular dependency") as exc_info:
        CircularTask([], error_fn=lambda x, y: abs(x - y))
    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_task_rejects_recursive_split_keys() -> None:
    r"""Recursive split-key lookup fails with the initialization diagnostic."""

    class RecursiveTask(DummyTask):
        @property
        def split_keys(self) -> tuple[int, ...]:
            return self.split_keys

    with pytest.raises(RuntimeError, match="circular dependency") as exc_info:
        RecursiveTask([], error_fn=lambda x, y: abs(x - y))
    assert isinstance(exc_info.value.__cause__, RecursionError)
