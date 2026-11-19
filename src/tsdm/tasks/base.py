r"""Protocols and base classes for reproducible time-series experiments.

A task combines a dataset with its partitions, sampling procedure, batch
construction, and evaluation metric. Error functions receive predictions first
and targets second; lower values indicate better predictions.

### Decomposable metrics

Mean squared error can be computed per sample and then averaged across the
dataset. The following example assumes equally weighted samples and finite
tensors of shape `(batch, ...)`:

```python
residuals = [
    (model(inputs) - targets).square().flatten(start_dim=1).mean(dim=1)
    for inputs, targets in test_loader
]
score = torch.cat(residuals).mean()
```

### Non-decomposable metrics

Metrics such as area under the receiver operating characteristic curve (AUROC)
must account for predictions across batches. Concatenate predictions and targets
before passing them to an appropriate metric implementation:

```python
outputs = [(model(inputs), targets) for inputs, targets in test_loader]
predictions = torch.cat([prediction for prediction, _ in outputs])
targets = torch.cat([target for _, target in outputs])
score = test_metric(predictions, targets)
```

These examples require a non-empty test loader. Metrics that support streaming
updates may accumulate their state instead of retaining all predictions.

### Encoders and data loaders

Encoders convert raw observations into model inputs. When several tensors
represent the same observations, their sample axes must remain aligned.
Fit preprocessing on the training partition and reuse it for the associated
validation and test partitions to avoid data leakage.

A sampler yields identifiers into a sample generator. The generator constructs
samples from a dataset partition, and the collate function combines those
samples into a batch. `TimeSeriesTask.make_dataloader` connects these components
using `torch.utils.data.DataLoader`.

### Flowchart sketch

```text
dataset → partitions → sample generator ← sampler
                            ↓
                          samples
                            ↓
                       collate function ← training-fitted encoder
                            ↓
                          batch → model → predictions
                            ↓                 ↓
                          targets → error function
```
"""

__all__ = [
    # Protocol & ABCs
    "Task",
    "SplitType",
    "TimeSeriesTask",
]

import logging
from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    SupportsFloat,
    runtime_checkable,
)

from torch.utils.data import DataLoader

from tsdm.pprint import pprint_repr
from tsdm.random.samplers import Sampler
from tsdm.types.protocols import SupportsGetItem
from tsdm.utils.lazydict import LazyDict


class SplitType(StrEnum):
    r"""Type of a split."""

    TRAIN = "train"
    TRAIN_VALIDATION = "trainval"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"

    @classmethod
    def _missing_(cls, value: object) -> SplitType | None:
        if not isinstance(value, str):
            return None

        match value.lower():
            case "train" | "training":
                return cls.TRAIN
            case "trainval" | "train_val" | "train_valid" | "train_validation":
                return cls.TRAIN_VALIDATION
            case "valid" | "validation" | "val":
                return cls.VALIDATION
            case "test" | "testing":
                return cls.TEST
            case "infer" | "inference":
                return cls.INFERENCE
            case _:
                return None


@runtime_checkable
class Task[
    SplitKey,  # e.g. tuple[FoldID, SplitID] like (0, "train")
    DatasetT = Any,
    SampleID = Any,  # e.g. observation and prediction window
    SampleT = Any,  # e.g. (inputs=Tensor, targets=Tensor)
    BatchT = list[SampleT],  # e.g. (inputs=Tensor, targets=Tensor)
    TargetT = Any,
](Protocol):
    r"""Protocol for tasks.

    Type Parameters:
        SplitKey: The key identifying a dataset partition, such as `"train"`,
            `(2, "test")`, or a longer tuple for nested cross-validation.
        DatasetT: The type of the raw dataset and its partitions.
        SampleID: The identifier yielded by a sampler and accepted by a generator.
        SampleT: The sample returned by a generator, such as a pair of inputs and targets.
        BatchT: The batch returned by a data loader.
        TargetT: The type of predictions and targets accepted by the error function.
    """

    @property  # the raw datase
    def dataset(self) -> DatasetT: ...

    @property  # e.g. the training data of the first fold
    def partitions(self) -> Mapping[SplitKey, DatasetT]: ...

    @property
    def generators(self) -> Mapping[SplitKey, SupportsGetItem[SampleID, SampleT]]: ...

    def make_dataloader[T](
        self,
        key: SplitKey,
        /,
        *,
        batch_size: int | None = None,
        drop_last: bool | None = None,
        collate_fn: Callable[[list[SampleT]], T] | None = None,
        **loader_kwargs,
    ) -> Iterable[T]: ...

    @property
    def error_fn(self) -> Callable[[TargetT, TargetT], SupportsFloat]:
        r"""Compare predictions and targets; lower is better.

        Arguments are ordered as `(predictions, targets)`.
        """
        ...


@pprint_repr
@dataclass  # (slots=True)
class TimeSeriesTask[
    SplitKey,  # e.g. tuple[FoldID, SplitID] like (0, "train")
    DatasetT = Any,
    SampleID = Any,
    SampleT = Any,
    BatchT = list[SampleT],
]:
    r"""Base class for time-series tasks.

    Type Parameters:
        SplitKey: The key identifying a dataset partition.
        DatasetT: The type of the raw dataset and its partitions.
        SampleID: The identifier yielded by a sampler and accepted by a generator.
        SampleT: The sample returned by a generator.
        BatchT: The batch produced by a collate function.

    A task provides the following components:

    - A dataset and partitions derived from it, such as training, validation, and
        test partitions, optionally organized into cross-validation folds.
    - An error function of the form `error_fn(predictions, targets)` that returns
        a scalar, with lower values indicating better predictions.
    - A sample generator, sampler, collate function, and data loader for each split.
        The sampler yields identifiers accepted by the generator, which returns
        samples containing context and target values.
    - Optional encoders and decoders that convert data into the representation
        required by the model or metric. Fit encoders on the training partition and
        reuse them for the associated validation and test partitions.

    By convention, a split key is a partition name or a sequence of the form
    `[*fold_ids, partition]`. The final element identifies the partition; preceding
    elements identify the cross-validation fold. Partition names should be
    recognized `SplitType` values or aliases. Subclasses implement `is_train_split`
    to identify their training partitions.

    Sample generators must support lookup by sample identifier. This base class
    does not support `torch.utils.data.IterableDataset` generators.
    """

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Class specific logger instance."""
    DEFAULT_TRAIN_BATCH_SIZE: ClassVar[int] = 32
    r"""Batch size for training splits."""
    DEFAULT_EVAL_BATCH_SIZE: ClassVar[int] = 64
    r"""Batch size for evaluation splits."""

    dataset: DatasetT
    r"""Dataset from which the splits are constructed."""

    _: KW_ONLY

    error_fn: Callable

    partitions: Mapping[SplitKey, DatasetT] = field(init=False)
    r"""Dataset partition associated with each split key."""
    collate_fns: Mapping[SplitKey, Callable[[list[SampleT]], BatchT]] = field(
        init=False
    )
    r"""Collate function used to create batches from samples."""
    dataloaders: Mapping[SplitKey, Iterable[BatchT]] = field(init=False)
    r"""Data loader associated with each split key."""
    generators: Mapping[SplitKey, SupportsGetItem[SampleID, SampleT]] = field(
        init=False
    )
    r"""Dictionary holding the sample generator associated with each key."""
    samplers: Mapping[SplitKey, Sampler[SampleID]] = field(init=False)
    r"""Sampler associated with each split key."""

    def __post_init__(self) -> None:
        r"""Initialize the task object."""
        message = (
            f"{type(self).__name__}.split_keys must provide a non-empty collection of split keys "
            "before the lazy mappings are initialized. It must not depend on partitions, "
            "generators, samplers, collate_fns, or dataloaders, as that creates a circular dependency."
        )
        try:
            keys = self.split_keys
        except (AttributeError, RecursionError) as exc:
            raise RuntimeError(message) from exc
        if not keys:
            raise RuntimeError(message)
        keys = tuple(keys)

        # create LazyDicts for the Mapping attributes
        self.collate_fns = LazyDict.from_func(self.make_collate_fn, keys)
        self.dataloaders = LazyDict.from_func(self.make_dataloader, keys)
        self.generators = LazyDict.from_func(self.make_generator, keys)
        self.samplers = LazyDict.from_func(self.make_sampler, keys)
        self.partitions = LazyDict.from_func(self.make_partition, keys)

    @property
    @abstractmethod
    def split_keys(self) -> Collection[SplitKey]:
        r"""Return the non-empty collection of unique split keys.

        Keys must remain stable throughout the task's lifetime. This property must
        be available before the lazy mappings exist and must not depend on them
        to avoid a circular dependency.
        """
        raise NotImplementedError

    @abstractmethod
    def is_train_split(self, key: SplitKey) -> bool:
        r"""Return whether the key is a training split."""
        raise NotImplementedError

    @abstractmethod
    def make_partition(self, key: SplitKey, /) -> DatasetT:
        r"""Return the sub-dataset associated with the specified split like `(0, "train")`."""
        raise NotImplementedError

    @abstractmethod
    def make_sampler(self, key: SplitKey, /) -> Sampler[SampleID]:
        r"""Create the sampler associated with the specified key."""
        return NotImplemented

    @abstractmethod
    def make_generator(self, key: SplitKey, /) -> SupportsGetItem[SampleID, SampleT]:
        r"""Return the sample generator associated with the specified key."""
        return NotImplemented

    def make_collate_fn(self, key: SplitKey, /) -> Callable[[list[SampleT]], BatchT]:  # ruff: ignore[ARG002]
        r"""Return the collate function which combines samples into a batch.

        Note:
            `collate_fn` is allowed to be key-dependent for cases when an encoder
            fitted on the training split is used to transform the validation/test splits.
        """
        return lambda x: x  # type: ignore

    def make_dataloader[T](
        self,
        key: SplitKey,
        /,
        *,
        batch_size: int | None = None,
        drop_last: bool | None = None,
        collate_fn: Callable[[list[SampleT]], T] | None = None,
        **loader_kwargs,
    ) -> Iterable[T]:
        r"""Return the dataloader associated with the specified key."""
        self.LOGGER.info("Creating DataLoader for key=%s", key)

        dataset = self.generators[key]

        kwargs: dict = {
            "batch_size": (
                batch_size
                if batch_size is not None
                else self.DEFAULT_TRAIN_BATCH_SIZE
                if self.is_train_split(key)
                else self.DEFAULT_EVAL_BATCH_SIZE
            ),
            "drop_last": (
                drop_last if drop_last is not None else self.is_train_split(key)
            ),
            "pin_memory": True,
            "sampler": self.samplers[key],
            "collate_fn": (
                collate_fn if collate_fn is not None else self.collate_fns[key]
            ),
        } | loader_kwargs

        return DataLoader(dataset, **kwargs)  # type: ignore


if TYPE_CHECKING:
    # ensure base classes are compatible with protocols
    # TODO: subclass protocols when PEP 767 (ReadOnly attributes) is accepted.

    def _upcast[SplitKey, DatasetT, SampleID, SampleT, BatchT](
        task: TimeSeriesTask[
            SplitKey,
            DatasetT,
            SampleID,
            SampleT,
            BatchT,
        ],
    ) -> Task[
        SplitKey,
        DatasetT,
        SampleID,
        SampleT,
        BatchT,
    ]:
        r"""Upcast a `TimeSeriesTask` to the `Task` protocol."""
        return task
