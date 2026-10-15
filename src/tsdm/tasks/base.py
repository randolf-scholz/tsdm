r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object. We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.


Forecasting Model
-----------------


Decomposable METRICS
--------------------

.. admonition:: **Example** Mean Square Error (MSE)

    .. code-block:: python

        dims = (1, ...)  # sum over all axes except batch dimension
        # y, yhat are of shape (B, ...)
        test_metric = lambda yhat, y: torch.sum((yhat - y) ** 2, dim=dims)
        accumulation = torch.mean

.. admonition:: **Recipe**

    .. code-block:: python

        r = []
        for x, y in dataloader:
            r.append(test_metric(model(x), y))

        score = accumulation(torch.concat(r, dim=BATCHDIM))

Non-decomposable METRICS
------------------------

.. admonition:: **Example** Area Under the Receiver Operating Characteristic Curve (AUROC)

    .. code-block:: python

        test_metric = (
            torch.AUROC()
        )  # expects two tensors of shape (N, ...) or (N, C, ...)
        score = test_metric([(model(x), y) for x, y in test_loader])
        # accumulation = None or identity function (tbd.)

.. admonition:: **Recipe**

    .. code-block:: python

        ys, yhats = []
        for x, y in dataloader:
            ys.append(y)
            yhats.append(model(x))

        ys = torch.concat(ys, dim=BATCHDIM)
        yhats = torch.concat(yhats, dim=BATCHDIM)
        score = test_metric(yhats, ys)

Normal Encoder
--------------

A normal pre_encoder is a pre_encoder with the property that all output tensors
share the same index axis.

I.e. it has a signature of the form ``list[tensor[n, ...]] -> list[tensor[n, ...]]``.
Pre-Encoder: Map DataFrame to torch.util.data.Dataset

Default DataLoader Creation
---------------------------

.. code-block:: python

    data = pre_processor.encode(data)  # `DataFrame` to `DataFrame`
    data = pre_encoder.encode(data)  # `DataFrame` to `DataSet`
    dataset = TensorDataset(*inputs, targets)
    sampler = SequenceSampler(tuple[TimeTensor], tuple[StaticTensor])
    dataloader = DataLoader(dataset, sampler=sampler, collate=...)
    batch = next(dataloader)

The whole responsibility of this class is to use a key from the "sampler" and
create a `Sample` from the `TimeSeriesCollection`.

**`Flowchart Sketch <image_link_>`_**

.. literalinclude:: ../content/diagrams/flowchart.txt
.. _image_link:
    https://asciiflow.com/#/share/eJzdWLtu2zAU%2FRWBUwtkcbu0XjwEqLcgQJJNQKDKTEtUL0gMEDcIYBgdO3AQXA0dM2bqWOhr9CWVKZEixYeo
    eKsgxBLJe3nv4bmHYh5BEsQQLJP7KDoDUbCFOViCRx88%2BGD58f3izAfb9undh%2BMThg%2B4ffGBZ7ua8mdT7txuYjYhvp84TXH425T7T2m0WcME5g
    FO86b8PXbHBxXtr9XzvvspnXOomCU3eeFd7Uxz4DBh1EdlCZpOtL9GMbyCOYLFeRpFMMQoTQQwnl%2B5SkOqdBp3aEZwcLzkfJpDrVmRYQxfk1kRE1ua
    xLtD2OZSxz4xIgEDSq2rLELYSMBhSGHiJkzCdANzPTtH6z9zCSojP3cKe19Ua208tTUGYmKqLhdbOU5R3%2Bjs5Krb0RxnEdt9LTTh6rOU2idU4Ni%2F
    Wq1UjwoWRDJQXixVMZ6PTXaoxZkoob%2FB7YKRva%2FFfpT4IloUQZxFcCFUCFHGmyvaKrWkw0MhbcXa57KFibK5DCrP2TfpwPwc4PCrx8W8g4NrySDe
    fUsnKnSQTnhEsZdtwnZvCDC8vRP2BtpxDZMizVkA5sT0eZkwdqmJCaSm0RadHceeX95IDGUolCI%2FL1j%2BUlVqR3eLcSHgq9rI0XH1UARUw0LjZeWX
    yUS3nZ52ufh8E6bZ1sOUQ4WHU299efPWyScF63%2B%2BhzyHJysybgI8y7R3oMhkLfg%2BddekpYKD%2FAvUfuhIt7h5ElMSokMuYpqEBu207aXTpdTZ
    sr9TcjSpR9Xg1A3aPhsmxAYxty1%2FF5Yk6xvYf1pa3Yk7BIYFvo0hzlHIt6M1U1R7QgYWOZOLSHnYQa6ml2rsesYWZTnsVPLmgJLsnhOetsQt4JHYkO
    Vwg%2Bh5jBeGWAC17FxPW5OjIVwXvlf6HkOzzl5cnFl1NkbXwuPDHwu7X3NiOMmWfnvqj6nqDBMC7BkG2f7fMQb1Oc0witF3el5sfv0Qu6K0KCbzVF1q
    iWKgig%2BewNM%2FqIfMMw%3D%3D)
"""

__all__ = [
    # Protocol & ABCs
    "Task",
    "SplitType",
    "TimeSeriesTask",
]

import logging
from abc import abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
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
from tsdm.timeseries import TimeSeriesCollection
from tsdm.types.protocols import SupportsGetItem, SupportsKeysAndGetItem
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
    DatasetT = Any,
    SplitKey = Any,  # e.g. tuple[FoldID, SplitID] like (0, "train")
    SampleID = Any,  # e.g. observation and prediction window
    SampleT = Any,  # e.g. (inputs=Tensor, targets=Tensor)
    BatchT = list[SampleT],  # e.g. (inputs=Tensor, targets=Tensor)
    TargetT = Any,
](Protocol):
    r"""Protocol for tasks.

    Type Parameters:
        DatasetT: The type of the raw dataset.
        SplitKey: The key to identify a split within a dataset, e.g.
            - A single SplitType like "train" or "test"
            - A tuple like (2, "test") for the test partition of the second fold
            - longer tuples when performing nested cross-validation.
        SampleID: The random input for the sampler.
        SampleT: The type of sample returned by the generator.
            (e.g. ``tuple[InputType, TargetType]``)
        BatchT: The batched data type returned by the dataloaders
    """

    @property  # the raw datase
    def dataset(self) -> DatasetT: ...

    @property  # e.g. the training data of the first fold
    def partitions(self) -> Mapping[SplitKey, DatasetT]: ...

    @property
    def generators(self) -> Mapping[SplitKey, SupportsGetItem[SampleID, SampleT]]: ...

    def make_dataloader[T = BatchT](
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
        """Compare predictions and targets; lower is better.

        Arguments are ordered as (predictions, targets).
        """
        ...


@pprint_repr
@dataclass  # (slots=True)
class TimeSeriesTask[
    SplitKey,  # e.g. tuple[FoldID, SplitID] like (0, "train")
    SampleID = Any,
    SampleT = Any,
    BatchT = list[SampleT],
]:
    r"""Abstract Base Class for Tasks.

    Type Parameters:
        SplitKey
        SampleID
        SampleT
        BatchT

    A task has the following responsibilities:

    - Provide a dataset
        - All splits should be derived from the same dataset
    - Provide splits of the dataset, usually one of the following:
        - train, test
        - train, validation, test
        - cross-validation + train/(valid)/test: (0, train), (0, test), ..., (n, train), (n, test)
        - nested cross-validation: (0, 0), (0, 1), ..., (0, n), (1, 0), ..., (n, n)
    - Provide a metric for evaluation
        - must be of the form ``Callable[[target, prediction], Scalar]``
    - Provide a `torch.utils.data.DataLoader` for each split
        - Provide a `torch.utils.data.Dataset` for each split
            - Dataset should return `Sample` objects providing context and target values.
        - Provide a `torch.utils.data.Sampler` for each split
            - Sampler should return indices into the dataset
    - Optional: Task specific encoder/decoder
        - Since we use raw format, such an encoder might be necessary to even meaningfully
          compute the target metric
        - For examples, tasks may ask for an MSE-loss on standardized data, or a cross-entropy for
          categorical data. In such cases, the task should provide an encoder/decoder to convert the
          data to the desired format.
        - For the simple case when both the input and the target are DataFrames, the task should
          provide DataFrameEncoders, one for each of the index / timeseries / metadata.
        - Encoders must be fit on the training data, but also transform the test data.
            - Need a way to associate a train split to each test/valid split.
            - CONVENTION: a split key is either a partition name or a sequence of the form
              ``[*fold_ids, partition]``. Only the final element identifies the partition; all
              preceding elements identify the cross-validation fold. The partition must be a
              recognized `SplitType` value or alias.

    To make this simpler, we first consider the `Mapping` interface,
    i.e. all the samplers are of fixed sized. and the dataset is a `Mapping` type.
    We do not support `torch.utils.data.IterableDataset`, as this point.
    """

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Class specific logger instance."""

    dataset: TimeSeriesCollection
    r"""Dataset from which the splits are constructed."""

    _: KW_ONLY

    partitions: SupportsKeysAndGetItem[SplitKey, Any] = field(init=False)
    r"""Dictionary holding `Fold` associated with each key (index for split)."""
    collate_fns: Mapping[SplitKey, Callable[[list[SampleT]], BatchT]] = field(
        init=False
    )
    r"""Collate function used to create batches from samples."""
    dataloaders: Mapping[SplitKey, DataLoader[BatchT]] = field(init=False)
    r"""Dictionary holding `DataLoader` associated with each key."""
    generators: Mapping[SplitKey, SupportsGetItem[SampleID, SampleT]] = field(
        init=False
    )
    r"""Dictionary holding the sample generator associated with each key."""
    samplers: Mapping[SplitKey, Sampler[SampleID]] = field(init=False)
    r"""Dictionary holding `Sampler` associated with each key."""

    train_batch_size: int = 32
    r"""Batch size for training splits."""
    eval_batch_size: int = 64
    r"""Batch size for evaluation splits."""

    validate: bool = True
    r"""Whether to validate the folds."""
    initialize: bool = True
    r"""Whether to initialize the task object."""

    def __post_init__(self) -> None:
        r"""Initialize the task object."""
        # check the folds for consistency
        if self.validate:
            self.validate_folds()

        if not self.initialize:
            return

        # create LazyDicts for the Mapping attributes
        self.collate_fns = LazyDict.from_func(self.make_collate_fn, self)
        self.dataloaders = LazyDict.from_func(self.make_dataloader, self)
        self.generators = LazyDict.from_func(self.make_generator, self)
        self.samplers = LazyDict.from_func(self.make_sampler, self)
        self.partitions = LazyDict.from_func(self.make_partition, self)

    @staticmethod
    def split_type(key: object, /) -> SplitType:
        r"""Return the split type encoded by a split key.

        By convention, a sequence key has the form ``[*fold_ids, partition]``.
        Only its final element is interpreted as the partition name.
        """
        match key:
            case str(partition):
                return SplitType(partition)
            case [*_, partition]:
                return SplitType(partition)
            case _:
                raise TypeError(f"Cannot infer split type from {key=}.")

    def __iter__(self) -> Iterator[SplitKey]:
        r"""Iterate over the split keys."""
        return iter(self.partitions.keys())

    def __len__(self) -> int:
        r"""Return the number of splits."""
        return len(self.partitions.keys())

    @abstractmethod
    def make_partition(self, key: SplitKey, /) -> TimeSeriesCollection:
        r"""Return the sub-dataset associated with the specified split like ``(0, "train")``."""
        return self.dataset[self.partitions[key]]

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
        return NotImplemented

    def make_dataloader(self, key: SplitKey, /, **opts: Any) -> DataLoader:
        r"""Return the dataloader associated with the specified key."""
        self.LOGGER.info("Creating DataLoader for key=%s", key)

        dataset = self.generators[key]

        kwargs: dict = {
            "batch_size": (
                self.train_batch_size
                if self.is_train_split(key)
                else self.eval_batch_size
            ),
            "drop_last": self.is_train_split(key),
            "pin_memory": True,
            "sampler": self.samplers[key],
            "collate_fn": (
                fn if (fn := self.collate_fns[key]) is not NotImplemented else None
            ),
        } | opts

        return DataLoader(dataset, **kwargs)  # type: ignore

    def is_train_split(self, key: SplitKey, /) -> bool:
        r"""Return whether the key is a training split."""
        return self.split_type(key) in {SplitType.TRAIN, SplitType.TRAIN_VALIDATION}

    def get_train_split(self, key: SplitKey, /) -> SplitKey:
        r"""Return the training split associated with the given split key.

        For example, given (3, "test"), return (3, "train") or (3, "trainval") depending on the folds.
        """
        target_key = (key,) if isinstance(key, str) else key
        if not isinstance(target_key, tuple) or not target_key:
            raise TypeError(f"Split keys must be strings or non-empty tuples: {key!r}.")

        if self.split_type(key) in {SplitType.TRAIN, SplitType.TRAIN_VALIDATION}:
            # fast path: if the key is already a training split, return it
            return key

        fold_keys = self.partitions.keys()
        if key not in fold_keys:
            raise KeyError(f"Unknown split key: {key!r}.")
        if not (
            all(isinstance(key, str) and key for key in fold_keys)
            or all(isinstance(key, tuple) and key for key in fold_keys)
        ):
            raise TypeError("Fold keys must be uniformly strings or non-empty tuples.")

        for fold_key in fold_keys:
            normalized_key = (fold_key,) if isinstance(fold_key, str) else fold_key
            assert isinstance(normalized_key, tuple)
            if (
                normalized_key[:-1] == target_key[:-1]
                and self.split_type(fold_key) == SplitType.TRAIN
            ):
                return fold_key

        raise KeyError(f"No training split found for {key!r}.")

    def validate_folds(self) -> None:
        r"""Make sure all keys are correct format `str` or ``tuple[*fold_ids, str]``.

        - If keys are `partition`, they are assumed to be `partition` keys.
        - If keys are ``*folds, partition``, then test whether for each fold there is a unique train partition.
        """
        fold_keys = self.partitions.keys()
        normalized_keys: set[tuple[Any, ...]]
        if all(isinstance(key, str) for key in fold_keys):
            normalized_keys = {(key,) for key in fold_keys}
        elif all(isinstance(key, tuple) and key for key in fold_keys):
            normalized_keys = {key for key in fold_keys if isinstance(key, tuple)}
        else:
            raise TypeError("Fold keys must be uniformly strings or non-empty tuples.")

        split_types = {key: self.split_type(key) for key in normalized_keys}
        fold_ids = {key[:-1] for key in normalized_keys}
        train_counts = Counter(
            key[:-1]
            for key, split_type in split_types.items()
            if split_type == SplitType.TRAIN
        )
        if not fold_ids or any(train_counts[fold_id] != 1 for fold_id in fold_ids):
            raise ValueError("Each fold must have a unique train partition.")


if TYPE_CHECKING:
    # ensure base classes are compatible with protocols
    # TODO: subclass protocols when PEP 767 (ReadOnly attributes) is accepted.

    def _upcast[SplitID, SampleID, SampleT, BatchT](
        task: TimeSeriesTask[SplitID, SampleID, SampleT, BatchT],
    ) -> Task[SplitID, SampleID, SampleT, BatchT]:
        r"""Upcast a TimeSeriesTask to a ForecastingTask."""
        return task
