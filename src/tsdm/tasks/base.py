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
        test_metric = lambda y, yhat: torch.sum((y - yhat) ** 2, dim=dims)
        accumulation = torch.mean

.. admonition:: **Recipe**

    .. code-block:: python

        r = []
        for x, y in dataloader:
            r.append(test_metric(y, model(x)))

        score = accumulation(torch.concat(r, dim=BATCHDIM))

Non-decomposable METRICS
------------------------

.. admonition:: **Example** Area Under the Receiver Operating Characteristic Curve (AUROC)

    .. code-block:: python

        test_metric = torch.AUROC()  # expects two tensors of shape (N, ...) or (N, C, ...)
        score = test_metric([(y, model(x)) for x, y in test_loader])
        # accumulation = None or identity function (tbd.)

.. admonition:: **Recipe**

    .. code-block:: python

        ys, yhats = []
        for x, y in dataloader:
            ys.append(y)
            yhats.append(model(x))

        ys = torch.concat(ys, dim=BATCHDIM)
        yhats = torch.concat(yhats, dim=BATCHDIM)
        score = test_metric(ys, yhats)

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
    # Protocol
    "ForecastingTask",
    # Classes
    "Batch",
    "Split",
    "TimeSeriesTask",
]

import logging
import warnings
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property

from pandas import DataFrame, Index, MultiIndex, Series
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from tsdm.data import (
    MapDataset,
    TimeSeriesCollection,
    TimeSeriesSampleGenerator,
    TorchDataset,
)
from tsdm.encoders import Encoder
from tsdm.metrics import Metric
from tsdm.random.samplers import Sampler
from tsdm.types.variables import key_var as K
from tsdm.utils import LazyDict
from tsdm.utils.strings import pprint_repr

Sample_co = TypeVar("Sample_co", covariant=True)
"""Covariant type variable for `Sample`."""

SplitID = TypeVar("SplitID", bound=Hashable)
"""Type of a split ID."""

Batch: TypeAlias = Tensor | Sequence[Tensor] | Mapping[str, Tensor]
"""Type of a batch of data."""


@dataclass
class Split(Generic[Sample_co]):
    """Represents a split of a dataset."""

    name: Hashable = NotImplemented
    r"""List of index."""
    fold: Series = NotImplemented
    r"""Dictionary holding `Fold` associated with each key (index for split)."""

    collate_fn: Callable[[list[Sample_co]], Batch] = NotImplemented
    r"""Collate function used to create batches from samples."""
    dataloader: DataLoader[Sample_co] = NotImplemented
    r"""Dictionary holding `DataLoader` associated with each key."""
    encoders: Encoder = NotImplemented
    r"""Dictionary holding `Encoder` associated with each key."""
    generator: TimeSeriesSampleGenerator = NotImplemented
    r"""Dictionary holding `torch.utils.data.Dataset` associated with each key."""
    sampler: Sampler = NotImplemented
    r"""Dictionary holding `Sampler` associated with each key."""
    split: TimeSeriesCollection = NotImplemented
    r"""Dictionary holding sampler associated with each key."""
    test_metric: Callable[[Tensor, Tensor], Tensor] = NotImplemented
    r"""Metric used for evaluation."""


@runtime_checkable
class ForecastingTask(Protocol[SplitID, K, Sample_co]):
    """Protocol for tasks."""

    @property
    def dataset(self) -> TimeSeriesCollection:
        """The dataset."""
        ...

    @property
    def folds(self) -> DataFrame:
        """A mapping of all the splits."""
        ...

    @property
    def splits(self) -> Mapping[SplitID, TimeSeriesCollection]:
        r"""Dictionary holding sampler associated with each key."""
        ...

    @property
    def samplers(self) -> Mapping[SplitID, Sampler[K]]:
        """A mapping of all the samplers."""
        ...

    @property
    def generators(self) -> Mapping[SplitID, MapDataset[K, Sample_co]]:
        r"""Dictionary holding the generator associated with each key."""
        ...

    @property
    def dataloaders(self) -> Mapping[SplitID, DataLoader[Sample_co]]:
        """A mapping of all the dataloaders."""
        ...

    @property
    def test_metrics(self) -> Mapping[SplitID, Metric]:
        """A mapping of all the test metrics."""
        ...


class TimeSeriesTaskMetaClass(type(Protocol)):  # type: ignore[misc]
    """Metaclass for TimeSeriesTask."""

    def __init__(
        self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)
        if not hasattr(self, "LOGGER"):
            self.LOGGER = logging.getLogger(f"{self.__module__}.{self.__name__}")


@pprint_repr
@dataclass
class TimeSeriesTask(Generic[SplitID, K, Sample_co], metaclass=TimeSeriesTaskMetaClass):
    r"""Abstract Base Class for Tasks.

    A task has the following responsibilities:

    - Provide a dataset
        - All splits should be derived from the same dataset
    - Provide splits of the dataset, usually one of the following:
        - train, test
        - train, validation, test
        - cross-validation + train/(valid)/test: (0, train), (0, test), ..., (n, train), (n, test)
        - nested cross-validation: (0, 0), (0, 1), ..., (0, n), (1, 0), ..., (n, n)
    - Provide a metric for evaluation
        - must be of the form `Callable[[target, prediction], Scalar]`
    - Provide a `torch.utils.data.DataLoader` for each split
        - Provide a `torch.utils.data.Dataset` for each split
            - Dataset should return `Sample` objects providing `Sample.Inputs` and `Sample.Targets`.
        - Provide a `torch.utils.data.Sampler` for each split
            - Sampler should return indices into the datase
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
            - ASSUMPTION: the split key is of the form `*fold, partition`, where `partition` is one of
              the strings `train`, `valid` or `test`, and `fold` is an integer or tuple of integers.

    To make this simpler, we first consider the `Mapping` interface,
    i.e. all the samplers are of fixed sized. and the dataset is a `Mapping` type.
    We do not support `torch.utils.data.IterableDataset`, as this point.

    .. note::

        - `torch.utils.data.Dataset[T_co]` implements
            - `__getitem__(self, index) -> T_co`
            - `__add__(self, other: Dataset[T_co]) -> ConcatDataset[T_co]`.
        - `torch.utils.data.Sampler[T_co]`
            - `__iter__(self) -> Iterator[T_co]`.
        - `torch.utils.data.DataLoader[T_co]`
            - attribute dataset: `Dataset[T_co]`
            - `__iter__(self) -> Iterator[Any]`
            - `__len__(self) -> int`.
    """

    # ABCs should have slots https://stackoverflow.com/a/62628857/9318372
    # __slots__ = ()

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""

    dataset: TimeSeriesCollection
    r"""Dataset from which the splits are constructed."""

    _: KW_ONLY

    index: Sequence[SplitID] = NotImplemented
    r"""List of index."""
    folds: DataFrame = NotImplemented
    r"""Dictionary holding `Fold` associated with each key (index for split)."""

    collate_fns: Mapping[SplitID, Callable[[list[Sample_co]], Batch]] = NotImplemented
    r"""Collate function used to create batches from samples."""
    dataloaders: Mapping[SplitID, DataLoader[Sample_co]] = NotImplemented
    r"""Dictionary holding `DataLoader` associated with each key."""
    encoders: Mapping[SplitID, Encoder] = NotImplemented
    r"""Dictionary holding `Encoder` associated with each key."""
    generators: Mapping[SplitID, MapDataset[K, Sample_co]] = NotImplemented
    r"""Dictionary holding `torch.utils.data.Dataset` associated with each key."""
    samplers: Mapping[SplitID, Sampler[K]] = NotImplemented
    r"""Dictionary holding `Sampler` associated with each key."""
    splits: Mapping[SplitID, TimeSeriesCollection] = NotImplemented
    r"""Dictionary holding sampler associated with each key."""
    test_metrics: Mapping[SplitID, Callable[[Tensor, Tensor], Tensor]] = NotImplemented
    r"""Metric used for evaluation."""

    train_patterns: Sequence[str] = ("train", "training")
    r"""List of patterns to match for training splits."""
    infer_patterns: Sequence[str] = ("test", "testing", "val", "valid", "validation")
    r"""List of patterns to match for infererence splits."""

    validate: bool = True
    """Whether to validate the folds."""
    initialize: bool = True

    def __post_init__(self) -> None:
        r"""Initialize the task object."""
        if self.folds is NotImplemented:
            self.LOGGER.info("No folds provided. Creating them.")
            self.folds = self.make_folds()

        # check the folds for consistency
        if self.validate:
            self.validate_folds()

        if self.index is NotImplemented:
            match self.folds:
                case DataFrame() as frame:
                    self.index = frame.columns
                case _:
                    self.index = Series(list(self.folds), name="folds")

        if not self.initialize:
            return

        # create LazyDicts for the Mapping attributes
        if self.collate_fns is NotImplemented:
            self.LOGGER.info("No collate functions provided. Caching them.")
            self.collate_fns = LazyDict({key: self.make_collate_fn for key in self})
        if self.dataloaders is NotImplemented:
            self.LOGGER.info("No DataLoaders provided. Caching them.")
            self.dataloaders = LazyDict({key: self.make_dataloader for key in self})
        if self.encoders is NotImplemented:
            self.LOGGER.info("No Encoders provided. Caching them.")
            self.encoders = LazyDict({key: self.make_encoder for key in self})
        if self.generators is NotImplemented:
            self.LOGGER.info("No Generators provided. Caching them.")
            self.generators = LazyDict({key: self.make_generator for key in self})
        if self.samplers is NotImplemented:
            self.LOGGER.info("No Samplers provided. Caching them.")
            self.samplers = LazyDict({key: self.make_sampler for key in self})
        if self.splits is NotImplemented:
            self.LOGGER.info("No splits provided. Creating them.")
            self.splits = LazyDict({key: self.make_split for key in self})
        if self.test_metrics is NotImplemented:
            self.LOGGER.info("No test metrics provided. Caching them.")
            self.test_metrics = LazyDict({key: self.make_test_metric for key in self})

    def __iter__(self) -> Iterator[SplitID]:
        r"""Iterate over the split keys."""
        return iter(self.index)

    def __len__(self) -> int:
        r"""Return the number of splits."""
        return len(self.index)

    @staticmethod
    def default_test_metric(*, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Return the test metric."""
        return NotImplemented

    @staticmethod
    def default_collate_fn(samples: list[Sample_co]) -> Batch:
        r"""Return the default collate function."""
        return samples  # type: ignore[return-value]

    def make_collate_fn(self, key: SplitID, /) -> Callable[[list[Sample_co]], Batch]:
        r"""Return the test metric."""
        return NotImplemented

    def make_dataloader(self, key: SplitID, /, **dataloader_kwargs: Any) -> DataLoader:
        r"""Return the dataloader associated with the specified key."""
        self.LOGGER.info("Creating DataLoader for key=%s", key)

        kwargs: dict = self.dataloader_config[key]
        dataset = self.generators[key]

        # sampler
        sampler = self.samplers[key]
        if sampler is NotImplemented:
            warnings.warn(f"No sampler provided for {key=}.", stacklevel=2)
            kwargs["sampler"] = None
        else:
            kwargs["sampler"] = sampler

        # collate_fn
        collate_fn = self.collate_fns[key]
        if collate_fn is NotImplemented:
            warnings.warn(f"No collate_fn provided for {key=}.", stacklevel=2)
            kwargs["collate_fn"] = self.default_collate_fn
        else:
            kwargs["collate_fn"] = collate_fn

        kwargs |= dataloader_kwargs
        return DataLoader(dataset, **kwargs)  # type: ignore[arg-type]

    def make_encoder(self, key: SplitID, /) -> Encoder:
        r"""Create the encoder associated with the specified key."""
        return NotImplemented

    @abstractmethod
    def make_folds(self, /) -> Mapping[SplitID, Series]:
        r"""Return the indices of the datapoints associated with the specific split."""
        return NotImplemented

    @abstractmethod
    def make_generator(self, key: SplitID, /) -> TorchDataset[K, Sample_co]:
        r"""Return the generator associated with the specified key."""
        return NotImplemented

    @abstractmethod
    def make_sampler(self, key: SplitID, /) -> Sampler[K]:
        r"""Create the sampler associated with the specified key."""
        return NotImplemented

    def make_split(self, key: SplitID, /) -> TimeSeriesCollection:
        r"""Return the sub-dataset associated with the specified split."""
        return self.dataset[self.folds[key]]

    def make_test_metric(self, key: SplitID, /) -> Callable[[Tensor, Tensor], Tensor]:
        r"""Return the test metric."""
        return NotImplemented

    def is_train_split(self, key: SplitID, /) -> bool:
        r"""Return whether the key is a training split."""
        match self.split_type(key):
            case "train":
                return True
            case "infer":
                return False
            case _:
                raise ValueError(f"Unknown split type for key={key}")

    def split_type(self, key: SplitID) -> Literal["train", "infer", "unknown"]:
        r"""Return the type of split."""
        if isinstance(key, str):
            if key.lower() in self.train_patterns:
                return "train"
            if key.lower() in self.infer_patterns:
                return "infer"
            return "unknown"
        if isinstance(key, Sequence):
            patterns = {self.split_type(k) for k in key}
            if patterns <= {"train", "unknown"}:
                return "train"
            if patterns <= {"infer", "unknown"}:
                return "infer"
            if patterns == {"unknown"}:
                return "unknown"
            raise ValueError(f"{key=} contains both train and infer splits.")
        return "unknown"

    @cached_property
    def dataloader_config(self) -> dict[SplitID, dict[str, Any]]:
        r"""Return dataloader configuration."""
        return {
            key: {
                "batch_size": 32,
                "drop_last": self.is_train_split(key),
                "pin_memory": True,
            }
            for key in self
        }

    @cached_property
    def train_split(self) -> Mapping[SplitID, SplitID]:
        r"""Return the matching train partition for the given key."""
        if isinstance(self.folds, Series | DataFrame):
            split_index = self.folds.T.index
        elif isinstance(self.folds, Mapping):
            split_index = Index(self.folds.keys())
        else:
            raise TypeError(f"Cannot infer train-partition from {type(self.folds)=}")
        if isinstance(split_index, MultiIndex):
            *fold, partition = names = split_index.names

            # Create Frame (fold, partition) -> (fold, partition.lower())
            df = split_index.to_frame()
            mask = df[partition].str.lower().isin(self.train_patterns)

            # create Series (train_key) -> train_key
            train_folds = df[mask].copy()
            train_folds = train_folds.drop(columns=names)
            train_folds["key"] = list(df[mask].index)
            train_folds = train_folds.droplevel(-1)

            # create Series (fold, partition) -> train_key
            df = df.drop(columns=names)
            df = df.join(train_folds, on=fold)
            return df["key"].to_dict()
        if isinstance(split_index, Index):
            mask = split_index.str.lower().isin(self.train_patterns)
            value = split_index[mask]
            s = Series(value.item(), index=split_index)
            return s.to_dict()
        raise RuntimeError("Supposed to be unreachable")

    def validate_folds(self) -> None:
        r"""Make sure all keys are correct format `str` or `tuple[str, ...]`.

        - If keys are `partition`, they are assumed to be `partition` keys.
        - If keys are `*folds, partition`, then test whether for each fold there is a unique train partition.
        """
        # get the index
        match self.folds:
            case Series() | DataFrame():
                split_index = self.folds.T.index
            case Mapping():
                split_index = Index(self.folds.keys())
            case _:
                raise TypeError(
                    f"Cannot infer train-partition from {type(self.folds)=}"
                )

        match split_index:
            case MultiIndex(names=names):
                *fold, partition = names
                df = split_index.to_frame(index=False)
                df["is_train"] = df[partition].str.lower().isin(self.train_patterns)
                if not all(df.groupby(fold)["is_train"].sum() == 1):
                    raise ValueError("Each fold must have a unique train partition.")
            case Index():
                mask = split_index.str.lower().isin(self.train_patterns)
                if not sum(mask) == 1:
                    raise ValueError("Each fold must have a unique train partition.")
            case _:
                raise TypeError(
                    f"Expected Index or MultiIndex, got {type(split_index)=}"
                )
