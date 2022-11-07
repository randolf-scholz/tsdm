r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.

Decomposable METRICS
--------------------

**Example:** Mean Square Error (MSE)

.. code-block:: python

    dims = (1, ...)  # sum over all axes except batch dimension
    # y, yhat are of shape (B, ...)
    test_metric = lambda y, yhat: torch.sum( (y-yhat)**2, dim=dims )
    accumulation = torch.mean

**Recipe**

.. code-block:: python

    r = []
    for x, y in dataloader:
        r.append( test_metric(y, model(x)) )

    score = accumulation(torch.concat(r, dim=BATCHDIM))

Non-decomposable METRICS
------------------------

**Example:** Area Under the Receiver Operating Characteristic Curve (AUROC)

test_metric = torch.AUROC()   # expects two tensors of shape (N, ...) or (N, C, ...)

.. code-block:: python

   score = test_metric([(y, model(x)) for x, y in test_loader])
   accumulation = None or identity function (tbd.)

**Recipe**

.. code-block:: python

    ys, yhats = []
    for x, y in dataloader:
        ys.append( y )
        yhats.append( model(x) )

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

    data = pre_processor.encode(data)  # DataFrame to DataFrame
    data = pre_encoder.encode(data)  # DataFrame to DataSet
    dataset = TensorDataset(*inputs, targets)
    sampler = SequenceSampler(tuple[TimeTensor], tuple[StaticTensor])
    dataloader = DataLoader(dataset, sampler=sampler, collate=....)
    batch = next(dataloader)

    inputs,
"""

from __future__ import annotations

__all__ = [
    # Classes
    "OldBaseTask",
    "TimeSeriesTaskDataset",
]

import logging
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

from pandas import NA, DataFrame, Index, Series
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler as TorchSampler

from tsdm.datasets import Dataset, TimeSeriesCollection, TimeSeriesDataset
from tsdm.encoders import ModularEncoder
from tsdm.utils import LazyDict
from tsdm.utils.strings import repr_dataclass, repr_namedtuple
from tsdm.utils.types import KeyVar

TimeSlice: TypeAlias = Index | slice | list

SampleType_co = TypeVar("SampleType_co", covariant=True)


class BaseTaskMetaClass(ABCMeta):
    r"""Metaclass for BaseTask."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


class OldBaseTask(ABC, Generic[KeyVar], metaclass=BaseTaskMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes
    ----------
    train_batch_size: int, default 32
        Default batch-size used by batchloader.
    eval_batch_size: int, default 128
        Default batch-size used by dataloaders (for evaluation).
    dataset: Dataset
        The attached dataset
    """

    # __slots__ = ()  # https://stackoverflow.com/a/62628857/9318372

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""
    preprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after batching)."""

    def __init__(self):
        r"""Initialize."""
        warnings.warn("deprecated, use new class", DeprecationWarning)

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        string = (
            f"{self.__class__.__name__}("
            # f"dataset={self.dataset.name}, "
            f"test_metric={type(self.test_metric).__name__})"
        )
        return string

    @property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DataFrame:
        r"""Return the cached dataset associated with the task."""

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyVar]:
        r"""List of index."""

    @property
    @abstractmethod
    def splits(self) -> Mapping[KeyVar, Any]:
        r"""Cache dictionary of dataset slices."""

    @abstractmethod
    def make_dataloader(
        self,
        key: KeyVar,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split.

        Args:
            key: From which part of the dataset to construct the loader.
            dataloader_kwargs: Options to be passed directly to the dataloader such as the generator.

        Returns:
            DataLoader: A DataLoader for the selected key.
        """

    @cached_property
    def dataloaders(self) -> Mapping[Any, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        kwargs: dict[Any, Any] = {
            # "key": key,
            "batch_size": self.eval_batch_size,
            "shuffle": False,
            "drop_last": False,
        }

        return LazyDict(
            {key: (self.make_dataloader, kwargs | {"key": key}) for key in self.splits}
        )


class BaseTask(ABC, Generic[KeyVar], metaclass=BaseTaskMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes
    ----------
    train_batch_size: int, default 32
        Default batch-size used by batchloader.
    eval_batch_size: int, default 128
        Default batch-size used by dataloaders (for evaluation).
    dataset: Dataset
        The attached dataset
    """

    # ABCs should have slots https://stackoverflow.com/a/62628857/9318372
    __slots__ = ()

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    preprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before sampling/batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after sampling/batching)."""

    dataloader_config_train = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    dataloader_config_infer = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    def __init__(
        self,
        dataset: TorchDataset,
        sampler: TorchSampler,
        dataloader_config_train: dict[str, Any],
        **dataloader_kwargs: Any,
    ):
        r"""Initialize the task object."""
        self.dataloader_config_train |= dataloader_kwargs
        self.dataloader_config_infer |= dataloader_kwargs

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        string = (
            f"{self.__class__.__name__}("
            # f"dataset={self.dataset.name}, "
            f"test_metric={type(self.test_metric).__name__})"
        )
        return string

    @property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DataFrame:
        r"""Return the cached dataset associated with the task."""

    def split_type(self, key: KeyVar) -> Literal["train", "infer", "unknown"]:
        r"""Return the type of split."""
        train_patterns = ["train", "training"]
        infer_patterns = [
            "test",
            "testing",
            "val",
            "valid",
            "validation",
            "eval",
            "evaluation",
            "infer",
            "inference",
        ]

        if isinstance(key, str):
            if key.lower() in train_patterns:
                return "train"
            if key.lower() in infer_patterns:
                return "infer"
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

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyVar]:
        r"""List of index."""

    @abstractmethod
    def make_split(self, key: KeyVar, /) -> TorchDataset:
        r"""Return the split associated with the given key."""

    @abstractmethod
    def make_sampler(self, key: KeyVar, /) -> TorchSampler:
        r"""Create sampler for the given split."""

    def make_dataloader(
        self,
        key: KeyVar,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split."""
        dataset = self.splits[key]
        sampler = self.samplers[key]

        match self.split_type(key):
            case "train":
                kwargs = self.dataloader_configs["train"]
            case "infer":
                kwargs = self.dataloader_configs["train"]
            case _:
                raise ValueError(f"Unknown split type: {self.split_type(key)=}")

        kwargs = kwargs | dataloader_kwargs
        dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        return dataloader

    @cached_property
    def dataloader_configs(self) -> dict[str, dict[str, Any]]:
        r"""Return dataloader configuration."""
        return {
            "train": self.dataloader_config_train,
            "eval": self.dataloader_config_infer,
        }

    @cached_property
    def splits(self) -> Mapping[KeyVar, TorchDataset]:
        r"""Cache dictionary of dataset splits."""
        return LazyDict({k: self.make_split for k in self.splits})

    @cached_property
    def samplers(self) -> Mapping[KeyVar, TorchSampler]:
        r"""Return a dictionary of samplers for each split."""
        return LazyDict({k: self.make_sampler for k in self.splits})

    @cached_property
    def dataloaders(self) -> Mapping[KeyVar, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        return LazyDict({k: self.make_dataloader for k in self.splits})


class EasyTask(ABC, Generic[KeyVar, SampleType_co], metaclass=BaseTaskMetaClass):
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
        - must be of the form `Callable[[target, prediction], Sclar]`
    - Provide a `torch.utils.data.DataLoader` for each split
        - Provide a `torch.utils.data.Dataset` for each split
            - Dataset should return `Sample` objects providing `Sample.Inputs` and `Sample.Targets`.
        - Provide a `torch.utils.data.Sampler` for each split
            - Sampler should return indices into the dataset

    To make this simpler, we here first consider the `Mapping` interface, i.e.
    Samplers are all of fixed sized. and the dataset is a `Mapping` type.
    We do not support `torch.utils.data.IterableDataset`, as this point.

    Note:
        - `torch.utils.data.Dataset[T_co]` implements `__getitem__` and `__add__`.
        - `torch.utils.data.Sampler[T_co]` implements `__iter__`.
        - `torch.utils.data.DataLoader[T_co]` implements `__iter__` and `__len__`.
    """

    # ABCs should have slots https://stackoverflow.com/a/62628857/9318372
    __slots__ = ()

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    preprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before sampling/batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after sampling/batching)."""

    dataloader_config_train = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    dataloader_config_infer = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    splits: Mapping[KeyVar, TorchDataset[SampleType_co]]
    samplers: Mapping[KeyVar, TorchSampler]

    def __init__(
        self,
        *,
        dataset: TorchDataset[SampleType_co],
        samplers: Mapping[KeyVar, TorchSampler],
        splits: Mapping[KeyVar, TorchDataset],
        test_metric: Callable[[Any, Any], Any],
        dataloader_config_train: Optional[dict[str, Any]] = None,
        dataloader_config_infer: Optional[dict[str, Any]] = None,
        dataloader_kwargs: dict[str, Any],
    ) -> None:
        r"""Initialize the task object."""
        if dataloader_config_train is not None:
            self.dataloader_config_train |= dataloader_config_train
        if dataloader_config_infer is not None:
            self.dataloader_config_infer |= dataloader_config_infer
        self.dataset = dataset
        self.splits = splits
        self.samplers = samplers

    @cached_property
    def dataloader_configs(self) -> dict[str, dict[str, Any]]:
        r"""Return dataloader configuration."""
        return {
            "train": self.dataloader_config_train,
            "eval": self.dataloader_config_infer,
        }

    def split_type(self, key: KeyVar) -> Literal["train", "infer", "unknown"]:
        r"""Return the type of split."""
        train_patterns = ["train", "training"]
        infer_patterns = [
            "test",
            "testing",
            "val",
            "valid",
            "validation",
            "eval",
            "evaluation",
            "infer",
            "inference",
        ]

        if isinstance(key, str):
            if key.lower() in train_patterns:
                return "train"
            if key.lower() in infer_patterns:
                return "infer"
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

    def make_dataloader(
        self,
        key: KeyVar,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split."""
        dataset = self.splits[key]
        sampler = self.samplers[key]

        match self.split_type(key):
            case "train":
                kwargs = self.dataloader_configs["train"]
            case "infer":
                kwargs = self.dataloader_configs["train"]
            case _:
                raise ValueError(f"Unknown split type: {self.split_type(key)=}")

        kwargs = kwargs | dataloader_kwargs
        dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        return dataloader


class Inputs(NamedTuple):
    r"""Tuple of inputs."""

    t_target: Series
    x: DataFrame
    u: Optional[DataFrame] = None
    metadata: Optional[DataFrame] = None

    def __repr__(self):
        return repr_namedtuple(self, recursive=False)


class Targets(NamedTuple):
    r"""Tuple of inputs."""

    y: DataFrame
    metadata: Optional[DataFrame] = None

    def __repr__(self):
        return repr_namedtuple(self, recursive=False)


class Sample(NamedTuple):
    r"""A sample for forecasting task."""

    key: Hashable
    inputs: Inputs
    targets: Targets

    def __repr__(self):
        return repr_namedtuple(self, recursive=1)

    def sparsify_index(self) -> Sample:
        r"""Drop rows that contain only NAN values."""
        if self.inputs.x is not None:
            self.inputs.x.dropna(how="all", inplace=True)

        if self.inputs.u is not None:
            self.inputs.u.dropna(how="all", inplace=True)

        if self.targets.y is not None:
            self.targets.y.dropna(how="all", inplace=True)

        if self.inputs.t_target is not None:
            diff = self.inputs.t_target.index.difference(self.targets.y.index)
            self.inputs.t_target.drop(diff, inplace=True)

        return self


TS_Type_co = TypeVar(
    "TS_Type_co", bound=TimeSeriesDataset | TimeSeriesCollection, covariant=True
)


@dataclass
class TimeSeriesTaskDataset(TorchDataset[Sample]):
    r"""Creates sample from a TimeSeriesCollection.

    There are different modus operandi for creating samples from a TimeSeriesCollection.

    Format Specification
    ~~~~~~~~~~~~~~~~~~~~

    - column-sparse
    - separate x and u and y

    - masked: In this format, two equimodal copies of the data are stored with appropriate masking.
        - inputs = (t, s, m)
        - targets = (s', m')

    - dense: Here, the data is split into groups of equal length. (x, u, y) share the same time index.
        - inputs = (t, x, u, m_x)
        - targets = (y, m_y)

    - sparse: Here, the data is split into groups sparse tensors. ALl NAN-only rows are dropped.
        - inputs = (t_y, (t_x, x), (t_u, u), m_x)
        - targets = (y, m_y)

    This class is used inside DataLoader.

    +---------------+------------------+------------------+
    | variable      | observation-mask | forecasting-mask |
    +===============+==================+==================+
    | observables X | ✔                | ✘                |
    +---------------+------------------+------------------+
    | controls U    | ✔                | ✔                |
    +---------------+------------------+------------------+
    | targets Y     | ✘                | ✔                |
    +---------------+------------------+------------------+

    Examples
    --------
    - time series classification task: empty forecasting horizon, only metadata_targets set.
    - time series imputation task: observation horizon and forecasting horizon overlap
    - time series forecasting task: observation horizon and forecasting horizon
    - time series forecasting task (autoregressive): observables = targets
    - time series event forecasting: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} given n
    - time series event forecasting++: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} and n

    Notes
    -----
    The option `pin_memory` of `torch.utils.data.DataLoader` recurses through
    Mappings and Sequence. However, it will cast the types. The only preserved Types are

    - dicts
    - tuples
    - namedtuples

    Dataclasses are currently not supported. Therefore, we preferably use namedtuples
    or dicts as containers.
    """

    dataset: TimeSeriesDataset | TimeSeriesCollection
    _: KW_ONLY = NotImplemented
    targets: Index
    r"""Columns of the data that are used as targets."""
    observables: Index = NotImplemented
    r"""Columns of the data that are used as inputs."""
    covariates: Optional[Index] = None
    r"""Columns of the data that are used as controls."""
    metadata_targets: Optional[Index] = None
    r"""Columns of the metadata that are targets."""
    metadata_observables: Optional[Index] = NotImplemented
    r"""Columns of the metadata that are targets."""
    sample_format: Union[
        Literal["masked", "sparse"],
        tuple[Literal["masked", "sparse"], Literal["masked", "sparse"]],
    ] = "masked"

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.observables is NotImplemented:
            self.observables = self.dataset.timeseries.columns
        if self.metadata_observables is NotImplemented:
            if self.dataset.metadata is None:
                self.metadata_observables = None
            else:
                self.metadata_observables = self.dataset.metadata.columns

    def __iter__(self) -> Iterator[TS_Type_co]:
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: KeyVar) -> Sample:
        match self.sample_format:
            case "masked" | ("masked", "masked"):
                return self.make_sample(key, sparse_index=False, sparse_columns=False)
            case ("sparse", "masked"):
                return self.make_sample(key, sparse_index=True, sparse_columns=False)
            case ("masked", "sparse"):
                return self.make_sample(key, sparse_index=False, sparse_columns=True)
            case "sparse" | ("sparse", "sparse"):
                return self.make_sample(key, sparse_index=True, sparse_columns=True)
            case _:
                raise ValueError(f"Unknown sample format {self.sample_format=}")

    def __repr__(self):
        return repr_dataclass(self, recursive=1)

    def make_sample(
        self, key: KeyVar, *, sparse_index: bool = False, sparse_columns: bool = False
    ) -> Sample:
        r"""Create a sample from a TimeSeriesCollection."""
        # extract key
        if isinstance(self.dataset, TimeSeriesDataset):
            assert isinstance(key, tuple) and len(key) == 2
            observation_horizon, forecasting_horizon = key
            tsd = self.dataset
        elif isinstance(self.dataset, TimeSeriesCollection):
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(key[1], Collection) and len(key[1]) == 2
            outer_key, (observation_horizon, forecasting_horizon) = key
            tsd = self.dataset[outer_key]
        else:
            raise NotImplementedError

        # timeseries
        ts_observed = tsd[observation_horizon]
        ts_forecast = tsd[forecasting_horizon]
        horizon_index = ts_observed.index.union(ts_forecast.index)
        ts = tsd[horizon_index]
        u: Optional[DataFrame] = None

        if sparse_columns:
            x = ts[self.observables].copy()
            x.loc[ts_forecast.index] = NA

            y = ts[self.targets].copy()
            y.loc[ts_observed.index] = NA

            if self.covariates is not None:
                u = ts[self.covariates].copy()
        else:
            x = ts.copy()
            # mask everything except covariates and observables
            columns = ts.columns.difference(self.covariates or [])
            x.loc[ts_observed.index, columns.difference(self.observables)] = NA
            x.loc[ts_forecast.index, columns] = NA

            y = ts.copy()
            # mask everything except targets in forecasting horizon
            y.loc[ts_observed.index] = NA
            y.loc[ts_forecast.index, ts.columns.difference(self.targets)] = NA

        # t_target
        t_target = y.index.to_series().copy()

        # metadata
        md = tsd.metadata
        md_targets: Optional[DataFrame] = None
        if self.metadata_targets is not None:
            assert md is not None
            md_targets = md[self.metadata_targets].copy()
            md = md.drop(columns=self.metadata_targets)

        # assemble sample
        inputs = Inputs(t_target=t_target, x=x, u=u, metadata=md)
        targets = Targets(y=y, metadata=md_targets)
        sample = Sample(key=key, inputs=inputs, targets=targets)

        if sparse_index:
            sample.sparsify_index()

        return sample
