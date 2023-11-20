"""Sample generators for time series data."""

__all__ = [
    # NamedTuples
    "Inputs",
    "Targets",
    "Sample",
    "PlainSample",
    "TimeSeriesSample",
    "PaddedBatch",
    # Functions
    "collate_timeseries",
    # Classes
    "TimeSeriesSampleGenerator",
    "FixedSliceSampleGenerator",
    "Split",
]


import warnings
from collections.abc import Callable, Collection, Hashable, Iterator, Sequence
from dataclasses import KW_ONLY, dataclass
from math import nan as NAN
from types import NotImplementedType

import numpy as np
import torch
from pandas import NA, DataFrame, Index, Series
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset
from typing_extensions import Any, Generic, NamedTuple, Optional, Self, assert_type

from tsdm.data.timeseries import TimeSeriesCollection, TimeSeriesDataset
from tsdm.encoders import Encoder
from tsdm.random.samplers import Sampler
from tsdm.tasks.base import Batch, Sample_co
from tsdm.types.variables import key_var as K
from tsdm.utils.strings import pprint_repr, repr_dataclass, repr_namedtuple

# region NamedTuples -------------------------------------------------------------------


class Inputs(NamedTuple):
    r"""Tuple of inputs."""

    q: Series
    """Query time points."""
    x: DataFrame
    """Observations"""
    u: Optional[DataFrame] = None
    """Covariates."""
    metadata: Optional[DataFrame] = None
    """Metadata."""

    def __repr__(self) -> str:
        return repr_namedtuple(self)


class Targets(NamedTuple):
    r"""Tuple of inputs."""

    y: DataFrame
    """Target values at the query times."""
    metadata: Optional[DataFrame] = None
    """Target metadata."""

    def __repr__(self) -> str:
        return repr_namedtuple(self)


class Sample(NamedTuple):
    r"""A sample for forecasting task."""

    key: Hashable
    """The key of the sample - e.g. tuple[outer_index, (obs_rane, forecasting_range)]."""
    inputs: Inputs
    """The predictors the model is allowed to base its forecast on."""
    targets: Targets
    """The targets the model is supposed to predict."""
    rawdata: Optional[Any] = None

    def __repr__(self) -> str:
        return repr_namedtuple(self)

    def sparsify_index(self) -> Self:
        r"""Drop rows that contain only NAN values."""
        if self.inputs.x is not None:
            self.inputs.x.dropna(how="all", inplace=True)

        if self.inputs.u is not None:
            self.inputs.u.dropna(how="all", inplace=True)

        if self.targets.y is not None:
            self.targets.y.dropna(how="all", inplace=True)

        if self.inputs.q is not None:
            diff = self.inputs.q.index.difference(self.targets.y.index)
            self.inputs.q.drop(diff, inplace=True)

        return self


@pprint_repr
class PlainSample(NamedTuple):
    r"""A single sample of the data."""

    key: Any
    t: Index
    x: DataFrame
    t_target: Index
    targets: DataFrame


@pprint_repr
class TimeSeriesSample(NamedTuple):
    r"""A single sample of time series data.

    Examples:
        time series forecasting/imputation::

            t, x, q, y = sample
            yhat = model(q, t, x)  # predict at time q given history (t, x).
            loss = d(y, yhat)  # compute the loss

        time series classification::

            t, x, *_, md_targets = sample
            md_hat = model(t, x)  # predict class of the time series.
            loss = d(md_targets, md_hat)  # compute the loss



    Attributes:
        t_inputs   (Float[Nᵢ]):   Timestamps of the inputs.
        inputs     (Float[Nᵢ×D]): The inputs for all timesteps.
        t_targets  (Float[Nₜ]):   Timestamps of the targets.
        targets    (Float[Nₜ×K]): The targets for the target timesteps.
        metadata   (Float[M]):    Static metadata.
        md_targets (Float[M]):    Static metadata targets.
    """

    t_inputs: Tensor
    inputs: Tensor
    t_targets: Tensor
    targets: Tensor
    metadata: Optional[Tensor] = None
    md_targets: Optional[Tensor] = None


@pprint_repr
class PaddedBatch(NamedTuple):
    r"""A padded batch of samples.

    Note:
        - B: batch-size
        - T: maximum sequence length (= N + K)
        - D: input dimension
        - K: target dimension

    Example:
        t, x, y, mq, mx, my = collate_timeseries(batch)
        yhat = model(t, x)  # predict for the whole padded batch
        r = where(my, y-yhat, 0)  # set residual to zero for missing values
        loss = r.abs().pow(2).sum()  # compute the loss

    Attributes:
        t (Float[B×T]):   Combined timestamps of inputs and targets.
        x (Float[B×T×D]): The inputs for all timesteps.
        y (Float[B×T×F]): The targets for the target timesteps.
        mq (Bool[B×T]):   The 'queries' mask, True if the given time stamp is a query.
        mx (Bool[B×T×D]): The 'inputs' mask, True indicates an observation, False a missing value.
        my (Bool[B×T×F]): The 'targets' mask, True indicates a target, False a missing value.
        metadata (Optional[Float[B×M]]):   Stacked metadata.
        md_targets (Optional[Float[B×M]]): Stacked metadata targets.

    In this context, 'query' means that the model should predict the value at this time stamp.
    Consequently, `mq` is identical to or-reducing `my` along the target dimension.
    """

    t: Tensor  # B×N:   the padded timestamps/queries.
    x: Tensor  # B×N×D: the padded input values.
    y: Tensor  # B×N×F: the padded target values.
    mq: Tensor  # B×N:   the 'queries' mask.
    mx: Tensor  # B×N×D: the 'inputs' mask.
    my: Tensor  # B×N×F: the 'targets' mask.
    metadata: Optional[Tensor] = None  # B×M:   stacked metadata.
    md_targets: Optional[Tensor] = None  # B×M:   stacked metadata targets.


# endregion NamedTuples ----------------------------------------------------------------


def collate_timeseries(batch: list[TimeSeriesSample]) -> PaddedBatch:
    r"""Collate timeseries samples into padded batch.

    Assumptions:
        - t_target is sorted.
    """
    masks_inputs: list[Tensor] = []
    masks_queries: list[Tensor] = []
    masks_target: list[Tensor] = []
    padded_inputs: list[Tensor] = []
    padded_queries: list[Tensor] = []
    padded_targets: list[Tensor] = []
    metadata: list[Tensor] = []
    md_targets: list[Tensor] = []

    for sample in batch:
        if sample.metadata is not None:
            metadata.append(sample.metadata)
        if sample.md_targets is not None:
            md_targets.append(sample.md_targets)

        t_inputs = sample.t_inputs
        x = sample.inputs
        t_target = sample.t_targets
        y = sample.targets

        # pad the x-values by the target length
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]),
            fill_value=NAN,
            device=x.device,
            dtype=x.dtype,
        )
        x_padded = torch.cat((x, x_padding))

        # get the whole time interval
        t_combined = torch.cat((t_inputs, t_target))
        sorted_idx = torch.argsort(t_combined)

        # create a mask for looking up the target values
        m_queries = torch.cat([
            torch.zeros_like(t_inputs, dtype=torch.bool),
            torch.ones_like(t_target, dtype=torch.bool),
        ])
        m_inputs = x.isfinite()
        m_targets = y.isfinite()

        # append to lists, ordering by time
        masks_inputs.append(m_inputs[sorted_idx])
        masks_queries.append(m_queries[sorted_idx])
        masks_target.append(m_targets)  # assuming t_target is sorted
        padded_inputs.append(x_padded[sorted_idx])
        padded_queries.append(t_combined[sorted_idx])
        padded_targets.append(y)  # assuming t_target is sorted

    return PaddedBatch(
        t=pad_sequence(padded_queries, batch_first=True).squeeze(),
        x=pad_sequence(padded_inputs, batch_first=True, padding_value=NAN).squeeze(),
        y=pad_sequence(padded_targets, batch_first=True, padding_value=NAN).squeeze(),
        mq=pad_sequence(masks_queries, batch_first=True).squeeze(),
        mx=pad_sequence(masks_inputs, batch_first=True).squeeze(),
        my=pad_sequence(masks_target, batch_first=True).squeeze(),
        metadata=torch.stack(metadata) if metadata else None,
        md_targets=torch.stack(md_targets) if md_targets else None,
    )


@dataclass
class TimeSeriesSampleGenerator(TorchDataset[Sample]):
    r"""Creates sample from a TimeSeriesCollection.

    This class is responsible for creating samples from a TimeSeriesCollection.
    It acts as a Map-Stype Dataset, keys should be created from an appriopriate Sampler instance.

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
    """The dataset to sample from."""

    _: KW_ONLY

    targets: Index | list = NotImplemented
    r"""Columns of the data that are used as targets."""
    observables: Index | list = NotImplemented
    r"""Columns of the data that are used as inputs."""
    covariates: Index | list = NotImplemented
    r"""Columns of the data that are used as controls."""
    metadata_targets: Optional[Index | list] = None
    r"""Columns of the metadata that are targets."""
    metadata_observables: Optional[Index | list] = NotImplemented
    r"""Columns of the metadata that are targets."""
    sparse_index: bool = False
    r"""Whether to drop sparse rows from the index."""
    sparse_columns: bool = False
    r"""Whether to drop sparse cols from the data."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.targets is NotImplemented:
            self.targets = []
        if self.observables is NotImplemented:
            self.observables = self.dataset.timeseries.columns
        if self.covariates is NotImplemented:
            self.covariates = []
        if self.metadata_observables is NotImplemented:
            if self.dataset.metadata is None:
                self.metadata_observables = None
            else:
                self.metadata_observables = self.dataset.metadata.columns
        self.validate()

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: K) -> Sample:
        return self.make_sample(
            key, sparse_index=self.sparse_index, sparse_columns=self.sparse_columns
        )

    def __repr__(self) -> str:
        return repr_dataclass(self)

    def get_subgenerator(self, key: K) -> Self:
        r"""Get a subgenerator."""
        other_kwargs = {k: v for k, v in self.__dict__.items() if k != "dataset"}
        # noinspection PyArgumentList
        return self.__class__(self.dataset[key], **other_kwargs)

    def make_sample(
        self, key: K, *, sparse_index: bool = False, sparse_columns: bool = False
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

        assert_type(tsd, TimeSeriesDataset)

        # NOTE: observation horizon and forecasting horizon might be given in different formats
        # (1) slices  (2) indices (3) boolean masks

        # NOTE: Currently there is a bug with pandas indexing using [pyarrow] timestamps.
        # This means only boolean masks are supported.
        # https://github.com/pandas-dev/pandas/issues/53644 (fixed)
        # https://github.com/pandas-dev/pandas/issues/53645 (closed)
        # https://github.com/pandas-dev/pandas/issues/53154
        # https://github.com/apache/arrow/issues/36047 (closed)

        # timeseries
        ts_observed: DataFrame = tsd[observation_horizon]
        ts_forecast: DataFrame = tsd[forecasting_horizon]
        joint_horizon_index = ts_observed.index.union(ts_forecast.index)

        # FIXME: this is a workaround for the bug above.
        # FIXME: ArrowNotImplementedError: Function 'is_in' has no kernel matching input types (duration[ns])

        # NOTE: Using numpy since isin is broken for pyarrow timestamps.
        joint_horizon_mask = np.isin(tsd.timeindex, joint_horizon_index)
        ts = tsd[joint_horizon_mask]
        ts_observed_mask = np.isin(ts.index, ts_observed.index)
        ts_forecast_mask = np.isin(ts.index, ts_forecast.index)

        u: Optional[DataFrame] = None

        if sparse_columns:
            x = ts[self.observables].copy()
            x.loc[ts_forecast_mask] = NA

            y = ts[self.targets].copy()
            y.loc[ts_observed_mask] = NA

            u = ts[self.covariates].copy()
        else:
            x = ts.copy()
            # mask everything except covariates and observables
            columns = ts.columns.difference(self.covariates)
            x.loc[ts_observed_mask, columns.difference(self.observables)] = NA
            x.loc[ts_forecast_mask, columns] = NA

            y = ts.copy()
            # mask everything except targets in the forecasting horizon
            y.loc[ts_observed_mask] = NA
            y.loc[ts_forecast_mask, ts.columns.difference(self.targets)] = NA

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
        inputs = Inputs(q=t_target, x=x, u=u, metadata=md)
        targets = Targets(y=y, metadata=md_targets)
        sample = Sample(key=key, inputs=inputs, targets=targets, rawdata=ts)

        if sparse_index:
            sample.sparsify_index()

        return sample

    def validate(self) -> None:
        r"""Validate that chosen columns are present."""
        ts = self.dataset.timeseries
        md = self.dataset.metadata
        observables = set(self.observables)
        targets = set(self.targets)
        covariates = set() if self.covariates is None else set(self.covariates)
        md_observables = (
            set(self.metadata_observables)
            if self.metadata_observables is not None
            else set()
        )
        md_targets = (
            set() if self.metadata_targets is None else set(self.metadata_targets)
        )
        ts_columns = set(ts.columns)
        md_columns = set() if md is None else set(md.columns)

        if cols := covariates - ts_columns:
            raise ValueError(f"Covariates {cols} not in found timeseries columns!")
        if cols := observables - ts_columns:
            raise ValueError(f"Observables {cols} not in found timeseries columns!")
        if cols := targets - ts_columns:
            raise ValueError(f"Targets {cols} not found in timeseries columns!")
        if cols := covariates & observables:
            raise ValueError(f"Covariates and observables not disjoint! {cols}.")
        if cols := ts_columns - (observables | targets | covariates):
            warnings.warn(f"Unused columns in timeseries: {cols}", stacklevel=2)

        if md is not None:
            if cols := md_observables - md_columns:
                raise ValueError(f"Observables {cols} not in found metadata columns!")
            if cols := md_targets - md_columns:
                raise ValueError(f"Targets {cols} not in found metadata columns!")


@pprint_repr
@dataclass
class FixedSliceSampleGenerator(TorchDataset[PlainSample]):
    """Utility class for generating samples from a fixed slice of a time series.

    Assumptions:
        - `data_source` is a multi-index dataframe whose innermost index is the time index.
        - `key` is always a tuple for the n-1 outermost indices.
        - For each key, we are only interested in a fixed slice of the time series.
    """

    data_source: DataFrame
    input_slice: slice = NotImplemented
    target_slice: slice = NotImplemented

    _: KW_ONLY

    observables: Sequence[Hashable] = NotImplemented
    """These columns are unmasked over the obs.-horizon in the input slice."""
    targets: Sequence[Hashable] = NotImplemented
    """These columns are unmasked over the pred.-horizon in the target slice."""
    covariates: Sequence[Hashable] = ()
    """These columns are unmasked over the whole inputs slice."""

    def __post_init__(self) -> None:
        # set the index
        self.index: Index = self.data_source.reset_index(level=-1).index.unique()
        self.columns: Index = self.data_source.columns

        match self.input_slice, self.target_slice:
            case NotImplementedType(), NotImplementedType():
                raise ValueError("Please specify slices for the input and target.")
            case NotImplementedType(), slice():
                self.input_slice = slice(None, self.target_slice.start)
            case slice(), NotImplementedType():
                self.target_slice = slice(self.input_slice.stop, None)
            case slice(), slice():
                pass
            case _:
                raise TypeError(
                    "Incorrect input types! Explected slices but got"
                    f" {type(self.input_slice)=} and {type(self.target_slice)=}."
                )

        # validate the slices
        if self.input_slice.start is not None:
            assert self.target_slice.start is not None
            assert self.input_slice.start <= self.target_slice.start
        if self.input_slice.stop is not None:
            assert self.target_slice.start is not None
            assert self.input_slice.stop <= self.target_slice.start
        if self.target_slice.stop is not None:
            assert self.input_slice.stop is not None
            assert self.input_slice.stop <= self.target_slice.stop

        # set the combined slice
        self.combined_slice = slice(self.input_slice.start, self.target_slice.stop)

        match self.observables, self.targets:
            case NotImplementedType(), NotImplementedType():
                print(
                    "No observables/targets specified. Assuming autoregressive case:"
                    " all columns are both inputs and targets."
                )
                self.observables = self.columns.copy()
                self.targets = self.columns.copy()
            case NotImplementedType(), Sequence():
                warnings.warn(
                    "Targets are specified, but not observables. Assuming remaining"
                    " columns are observables.",
                    stacklevel=2,
                )
                self.observables = self.columns.difference(self.targets).copy()
            case Sequence(), NotImplementedType():
                warnings.warn(
                    "Observables are specified, but not targets. Assuming remaining"
                    " columns are targets.",
                    stacklevel=2,
                )
                self.targets = self.columns.difference(self.observables).copy()
            case Sequence(), Sequence():
                pass
            case _:
                raise TypeError(
                    "Incorrect input types! Explected sequences but got"
                    f" {type(self.observables)=} and {type(self.targets)=}."
                )

        # cast to index.
        self.observables = Index(self.observables)
        self.targets = Index(self.targets)
        self.covariates = Index(self.covariates)

        # validate the columns
        if not set(self.observables).issubset(self.columns):
            raise ValueError(
                "Observables contain columns that are not in the data source."
            )
        if not set(self.targets).issubset(self.columns):
            raise ValueError("Targets contain columns that are not in the data source.")

    def __len__(self) -> int:
        """Number of unique entries in the outer n-1 index levels."""
        return len(self.index)

    def __iter__(self) -> Iterator[PlainSample]:
        """Iterate over all the samples in the dataset."""
        for key in self.index:
            yield self[key]

    def __getitem__(self, key: Hashable) -> PlainSample:
        """Yield a single sample."""
        # select the individual time series
        ts = self.data_source.loc[key]
        # get the slices
        inputs = ts.loc[self.combined_slice]
        targets = ts.loc[self.target_slice]

        # get the  masks
        not_observed = self.columns.intersection(self.observables).union(
            self.columns.intersection(self.covariates)
        )
        not_targeted = self.columns.intersection(self.targets)
        not_covariates = self.columns.difference(self.covariates)

        # apply the masks
        inputs.loc[:, not_observed] = NAN
        inputs.loc[self.input_slice, not_covariates] = NAN
        targets.loc[:, not_targeted] = NAN

        return PlainSample(
            key=key,
            t=inputs.index.copy(),
            x=inputs.copy(),
            t_target=targets.index.copy(),
            targets=targets,
        )


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
