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

__all__ = [
    # Classes
    "BaseTask",
]

import logging
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import Any, ClassVar, Generic, Literal, NamedTuple, Optional

from pandas import NA, DataFrame, Index
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as TorchDataset

from tsdm.datasets import Dataset, TimeSeriesCollection
from tsdm.encoders import ModularEncoder
from tsdm.utils import LazyDict
from tsdm.utils.strings import repr_namedtuple
from tsdm.utils.types import KeyVar


class BaseTaskMetaClass(ABCMeta):
    r"""Metaclass for BaseTask."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


class BaseTask(ABC, Generic[KeyVar], metaclass=BaseTaskMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes
    ----------
    index: list[str]
        A list of string specifying the data splits of interest.
    train_batch_size: int, default 32
        Default batch-size used by batchloader.
    eval_batch_size: int, default 128
        Default batch-size used by dataloaders (for evaluation).
    preprocessor: Optional[Encoder], default None
        Task specific preprocessing. For example, the EVP might specifically ask for
        evaluation of Mean Squared Error on standardized data.
    dataset: Dataset
        The attached dataset
    splits: Mapping[KeyType, Any]
        Contains slices of the dataset. Contains a slice for each key, but may
        also hold additional entries. (For example: "joint" = "train"+"valid")
    batchloaders: Mapping[KeyType, DataLoader]
        The main DataLoader to be used for training models.
    dataloaders: Mapping[KeyType, DataLoader]
        Holds `DataLoaders` for all the index.
    """

    # __slots__ = ()  # https://stackoverflow.com/a/62628857/9318372

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""
    encoder: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after batching)."""

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
    def get_dataloader(
        self,
        key: KeyVar,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split.

        Parameters
        ----------
        key: str
            From which part of the dataset to construct the loader
        dataloader_kwargs:
            Options to be passed directly to the dataloader such as the generator.

        Returns
        -------
        DataLoader
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
            {key: (self.get_dataloader, kwargs | {"key": key}) for key in self.splits}
        )


class Inputs(NamedTuple):
    r"""Tuple of inputs."""

    t_target: Any
    x: Any
    u: Optional[Any] = None
    metadata: Optional[Any] = None

    def __repr__(self):
        return repr_namedtuple(self, recursive=False)


class Targets(NamedTuple):
    r"""Tuple of inputs."""

    y: Any
    metadata: Any

    def __repr__(self):
        return repr_namedtuple(self, recursive=False)


class Sample(NamedTuple):
    r"""A sample for forecasting task."""

    key: int
    inputs: Inputs
    targets: int
    # originals: int

    def __repr__(self):
        return repr_namedtuple(self, recursive=False)


@dataclass
class TimeSeriesCollectionTask(TorchDataset):
    r"""Creates sample from a TimeSeriesCollection.

    There are different modus operandi for creating samples from a TimeSeriesCollection.


    Format Specification
    ~~~~~~~~~~~~~~~~~~~~

    - masked: In this format, two equimodal copies of the data are stored with appropriate masking.
        - inputs = (t, s, m)
        - targets = (s', m')

    - split: Here, the data is split into groups of equal length. (x, u, y) share the same time index.
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

    Technical Remark
    ----------------
    The option `pin_memory` of `torch.utils.data.DataLoader` recurses through
    Mappings and Sequence. However, it will cast the types. The only preserved Types are

    - dicts
    - tuples
    - namedtuples

    Dataclasses are currently not supported. Therefore, we preferably use namedtuples
    or dicts as containers.
    """

    dataset: TimeSeriesCollection
    _: KW_ONLY = NotImplemented
    targets: Index
    r"""Columns of the data that are used as targets."""
    observables: Index = NotImplemented
    r"""Columns of the data that are used as inputs."""
    covariates: Optional[Index] = None
    r"""Columns of the data that are used as controls."""
    metadata_targets: Optional[Index] = None
    r"""Columns of the metadata that are targets."""
    metadata_observables: Optional[Index] = None
    r"""Columns of the metadata that are targets."""
    sample_format: Literal["masked", "split", "sparse"] = "masked"
    r"""Format of the samples."""

    def __post_init__(self):
        r"""Post init."""
        if self.observables is NotImplemented:
            self.observables = self.dataset.timeseries.columns

    def __getitem__(self, key: Any) -> Sample:
        assert isinstance(key, tuple) and len(key) == 2
        outer_key, inner_key = key
        assert isinstance(inner_key, list) and len(inner_key) == 2
        observation_horizon, forecasting_horizon = inner_key

        # horizon = observation_horizon | forecasting_horizon

        tsd = self.dataset[outer_key]

        ts_observed = tsd[observation_horizon]
        ts_forecast = tsd[forecasting_horizon]

        x = ts_observed[self.observables]
        u = ts_observed[self.covariates]
        y = ts_forecast[self.targets]

        # metadata
        md = tsd.metadata
        if self.metadata_targets is not None:
            md_targets = md[self.metadata_targets]
            md = md.drop(columns=self.metadata_targets)
        else:
            md_targets = None

        return Sample(
            key=outer_key,
            inputs=Inputs(metadata=md),
            # targets=pre,
        )

    def _make_sparse_sample(
        self, key, observation_horizon, forecasting_horizon
    ) -> Sample:
        tsd = self.dataset[key]

        # timeseries
        ts_observed = tsd[observation_horizon]
        ts_forecast = tsd[forecasting_horizon]
        x = ts_observed[self.observables].dropna(how="all")
        y = ts_forecast[self.targets].dropna(how="all")
        t_target = y.index

        u: Optional[DataFrame] = None
        if self.covariates is not None:
            u = ts_observed[self.covariates].dropna(how="all")

        # metadata
        md = tsd.metadata
        md_targets: Optional[DataFrame] = None
        if self.metadata_targets is not None:
            md_targets = md[self.metadata_targets]
            md[
                md.columns.difference(self.metadata_targets)
                if self.metadata_observables is None
                else self.metadata_observables
            ] = NA

            # md = md.drop(columns=self.metadata_targets)

        inputs = Inputs(t_target=t_target, x=x, u=u, metadata=md)
        targets = Targets(y=y, metadata=md_targets) if md_targets is not None else y
        return Sample(key=key, inputs=inputs, targets=targets)

    def _make_masked_sample(
        self, key, observation_horizon, forecasting_horizon
    ) -> Sample:
        tsd = self.dataset[key]

        # timeseries
        ts_observed = tsd[observation_horizon]
        ts_forecast = tsd[forecasting_horizon]
        horizon = ts_observed.index.union(ts_forecast.index)
        ts = tsd[horizon]
        columns = ts.columns

        x = ts.copy()
        # mask everything except covariates and observables
        x.loc[
            observation_horizon,
            columns.difference(self.covariates.union(self.observables)),
        ] = NA
        x.loc[forecasting_horizon, columns.difference(self.covariates)] = NA

        y = ts.copy()
        # mask everything except targets in forecasting horizon
        y.loc[observation_horizon] = NA
        y.loc[forecasting_horizon, columns.difference(self.targets)] = NA

        # metadata
        md = tsd.metadata
        md_targets: Optional[DataFrame] = None
        if self.metadata_targets is not None:
            md_targets = md[self.metadata_targets]
            md = md.drop(columns=self.metadata_targets)

        # inputs = Inputs(t_target=t_target, x=x, u=u, metadata=md)
        # targets = Targets(y=y, metadata=md_targets) if md_targets is not None else y
        # return Sample(key=key, inputs=inputs, targets=targets)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
