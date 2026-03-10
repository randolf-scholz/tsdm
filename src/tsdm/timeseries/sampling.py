r"""Infrastructure for sampling time series data."""

__all__ = [
    # NamedTuples
    "Inputs",
    "Targets",
    "Sample",
    "PlainSample",
    # Classes
    "TimeSeriesSampleGenerator",
    "FixedSliceSampleGenerator",
]


import warnings
from collections.abc import Iterator, Sequence
from dataclasses import KW_ONLY, dataclass
from math import nan as NAN
from typing import Any, NamedTuple, Optional, Self

from pandas import NA, DataFrame, Index, Series

from tsdm import constants as const
from tsdm.constants import UNDEFINED
from tsdm.datatools import TorchDataset
from tsdm.timeseries.pandas import PandasTS, PandasTSC
from tsdm.utils.decorators import pprint_repr

type Key = Any
r"""Placeholder for the key type."""


@pprint_repr
class Inputs(NamedTuple):
    r"""Tuple of inputs."""

    q: Series
    r"""Query time points."""
    x: DataFrame
    r"""Observations"""
    u: Optional[DataFrame] = None
    r"""Covariates."""
    metadata: Optional[DataFrame] = None
    r"""Metadata."""


@pprint_repr
class Targets(NamedTuple):
    r"""Tuple of inputs."""

    y: DataFrame
    r"""Target values at the query times."""
    metadata: Optional[DataFrame] = None
    r"""Target metadata."""


@pprint_repr
class Sample(NamedTuple):
    r"""A sample for forecasting task."""

    key: Key
    r"""The key of the sample - e.g. tuple[outer_index, (obs_rane, forecasting_range)]."""
    inputs: Inputs
    r"""The predictors the model is allowed to base its forecast on."""
    targets: Targets
    r"""The targets the model is supposed to predict."""
    rawdata: Optional[Any] = None

    def drop_null_rows(self) -> Self:
        r"""Drop rows that contain only NAN values."""
        if self.inputs.x is not None:
            self.inputs.x.dropna(how="all", inplace=True)

        if self.inputs.u is not None:
            self.inputs.u.dropna(how="all", inplace=True)

        if self.targets.y is not None:
            self.targets.y.dropna(how="all", inplace=True)

        if self.inputs.q is not None and self.targets.y is not None:
            # drop all queries that are no longer in the target index
            missing = self.inputs.q.index.difference(self.targets.y.index)
            self.inputs.q.drop(missing, inplace=True)

        return self


@pprint_repr
@dataclass
class TimeSeriesSampleGenerator(TorchDataset[Any, Sample]):
    r"""Creates sample from a TimeSeriesCollection.

    This class is responsible for creating samples from a TimeSeriesCollection.
    It acts as a Map-Stype Dataset, keys should be created from an appriopriate Sampler instance.

    There are different modus operandi for creating samples from a TimeSeriesCollection.

    Note:
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

    Examples:
        - time series classification task: empty forecasting horizon, only metadata_targets set.
        - time series imputation task: observation horizon and forecasting horizon overlap
        - time series forecasting task: observation horizon and forecasting horizon
        - time series forecasting task (autoregressive): observables = targets
        - time series event forecasting: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} given n
        - time series event forecasting++: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} and n

    Notes:
        The option `pin_memory` of `torch.utils.data.DataLoader` recurses through
        Mappings and Sequence. However, it will cast the types. The only preserved Types are

        - dicts
        - tuples
        - namedtuples

    Dataclasses are currently not supported. Therefore, we preferably use namedtuples
    or dicts as containers.
    """

    dataset: PandasTS | PandasTSC
    r"""The dataset to sample from."""

    _: KW_ONLY

    targets: Index | list = UNDEFINED
    r"""Columns of the data that are used as targets."""
    observables: Index | list = UNDEFINED
    r"""Columns of the data that are used as inputs."""
    covariates: Index | list = UNDEFINED
    r"""Columns of the data that are used as controls."""
    metadata_targets: Optional[Index | list] = None
    r"""Columns of the metadata that are targets."""
    metadata_observables: Optional[Index | list] = UNDEFINED
    r"""Columns of the metadata that are targets."""
    sparse_index: bool = False
    r"""Whether to drop sparse rows from the index."""
    sparse_columns: bool = False
    r"""Whether to drop sparse cols from the data."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.targets is UNDEFINED:
            self.targets = []
        if self.observables is UNDEFINED:
            self.observables = self.dataset.timeseries.columns
        if self.covariates is UNDEFINED:
            self.covariates = []
        if self.metadata_observables is UNDEFINED:
            if self.dataset.static_covariates is None:
                self.metadata_observables = None
            else:
                self.metadata_observables = self.dataset.static_covariates.columns
        self.validate()

    def __getitem__(self, key: Key, /) -> Sample:
        return self.make_sample(
            key,
            sparse_index=self.sparse_index,
            sparse_columns=self.sparse_columns,
        )

    def get_subgenerator(self, key: Key) -> Self:
        r"""Get a subgenerator."""
        other_kwargs = {k: v for k, v in self.__dict__.items() if k != "dataset"}
        # noinspection PyArgumentList
        return self.__class__(self.dataset[key], **other_kwargs)

    def make_sample(
        self,
        key: Key,
        *,
        sparse_index: bool = False,
        sparse_columns: bool = False,
    ) -> Sample:
        r"""Create a sample from a TimeSeriesCollection.

        Args:
            key: The key of the sample - e.g. tuple[outer_index, (obs_rane, forecasting_range)].
            sparse_index: Whether to drop rows that contain only NAN values.
            sparse_columns: Whether to drop columns that contain only NAN values.

        """
        # extract key
        match self.dataset, key:
            case PandasTS() as tsd, [observation_horizon, forecasting_horizon]:
                pass
            case PandasTSC() as tsc, [
                outer_key,
                [observation_horizon, forecasting_horizon],
            ]:
                tsd = tsc[outer_key]
            case PandasTS() | PandasTSC(), _:
                raise ValueError(f"Invalid key: {key!r}")
            case _:
                raise TypeError(f"Invalid dataset type: {type(self.dataset)=}")

        assert isinstance(tsd, PandasTS)

        # extract the relevant time series and their joint index
        ts_observed: DataFrame = tsd[observation_horizon].timeseries
        ts_forecast: DataFrame = tsd[forecasting_horizon].timeseries
        joint_horizon_index = ts_observed.index.union(ts_forecast.index)

        joint_horizon_mask = tsd.timeindex.isin(joint_horizon_index)
        ts = tsd[joint_horizon_mask].timeseries
        ts_observed_mask = ts.index.isin(ts_observed.index)
        ts_forecast_mask = ts.index.isin(ts_forecast.index)

        if sparse_columns:
            x = ts[self.observables].copy()
            y = ts[self.targets].copy()
            covariates = ts[self.covariates].copy()

            x.loc[ts_forecast_mask] = NA
            y.loc[ts_observed_mask] = NA

        else:
            x = ts.copy()
            y = ts.copy()
            covariates = None

            # SEC: mask everything except covariates and observables
            non_covariate_mask = ts.columns.difference(self.covariates)
            non_predictor_mask = ts.columns.difference(
                self.observables + self.covariates
            )
            x.loc[ts_observed_mask, non_predictor_mask] = NA
            x.loc[ts_forecast_mask, non_covariate_mask] = NA

            # SEC: mask everything except targets in the forecasting horizon
            non_target_mask = ts.columns.difference(self.targets)
            y.loc[ts_observed_mask] = NA
            y.loc[ts_forecast_mask, non_target_mask] = NA

        t_target = y.index.to_series().copy()

        # metadata
        md = tsd.static_covariates
        md_targets: Optional[DataFrame] = None
        if self.metadata_targets is not None:
            if md is None:
                raise ValueError("Metadata targets specified but no metadata found.")
            md_targets = md[self.metadata_targets].copy()
            md = md.drop(columns=self.metadata_targets)

        # assemble sample
        inputs = Inputs(q=t_target, x=x, u=covariates, metadata=md)
        targets = Targets(y=y, metadata=md_targets)
        sample = Sample(key=key, inputs=inputs, targets=targets, rawdata=ts)

        if sparse_index:
            sample.drop_null_rows()

        return sample

    def validate(self) -> None:
        r"""Validate that chosen columns are present."""
        ts = self.dataset.timeseries
        md = self.dataset.static_covariates
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
class PlainSample(NamedTuple):
    r"""A single sample of the data."""

    key: Any
    t: Index
    x: DataFrame
    t_target: Index
    targets: DataFrame


@pprint_repr
@dataclass
class FixedSliceSampleGenerator(TorchDataset[Any, PlainSample]):
    r"""Utility class for generating samples from a fixed slice of a time series.

    Assumptions:
        - `data_source` is a multi-index dataframe whose innermost index is the time index.
        - `key` is always a tuple for the n-1 outermost indices.
        - For each key, we are only interested in a fixed slice of the time series.
    """

    data_source: DataFrame
    input_slice: slice = UNDEFINED
    target_slice: slice = UNDEFINED

    _: KW_ONLY

    observables: Sequence[Key] = UNDEFINED
    r"""These columns are unmasked over the obs.-horizon in the input slice."""
    targets: Sequence[Key] = UNDEFINED
    r"""These columns are unmasked over the pred.-horizon in the target slice."""
    covariates: Sequence[Key] = ()
    r"""These columns are unmasked over the whole inputs slice."""

    def __post_init__(self) -> None:
        # set the index
        self.index: Index = self.data_source.reset_index(level=-1).index.unique()
        self.columns: Index = self.data_source.columns

        match self.input_slice, self.target_slice:
            case const.UNDEFINED, const.UNDEFINED:
                raise ValueError("Please specify slices for the input and target.")
            case const.UNDEFINED, slice():
                self.input_slice = slice(None, self.target_slice.start)
            case slice(), const.UNDEFINED:
                self.target_slice = slice(self.input_slice.stop, None)
            case slice(), slice():
                pass
            case _:
                raise TypeError(
                    "Incorrect input types! Explected slices but got"
                    f" {type(self.input_slice)=} and {type(self.target_slice)=}."
                )

        # validate the slices
        self._validate_slices()

        # set the combined slice
        self.combined_slice = slice(self.input_slice.start, self.target_slice.stop)

        match self.observables, self.targets:
            case const.UNDEFINED, const.UNDEFINED:
                warnings.warn(
                    "No observables/targets specified. Assuming autoregressive case:"
                    " all columns are both inputs and targets.",
                    stacklevel=2,
                )
                self.observables = self.columns.copy()
                self.targets = self.columns.copy()
            case const.UNDEFINED, Sequence():
                warnings.warn(
                    "Targets are specified, but not observables. Assuming remaining"
                    " columns are observables.",
                    stacklevel=2,
                )
                self.observables = self.columns.difference(self.targets).copy()
            case Sequence(), const.UNDEFINED:
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

    def _validate_slices(self) -> None:
        if self.input_slice.start is not None:
            if self.target_slice.start is None:
                raise ValueError("Target slice must be specified, given input slice.")
            if self.input_slice.start > self.target_slice.start:
                raise ValueError("Input slice start must be before target slice start.")
        if self.input_slice.stop is not None:
            if self.target_slice.start is None:
                raise ValueError("Target slice must be specified, given input slice.")
            if self.input_slice.stop > self.target_slice.start:
                raise ValueError("Input slice stop must be before target slice start.")
        if self.target_slice.stop is not None:
            if self.input_slice.stop is None:
                raise ValueError("Input slice must be specified, given target slice.")
            if self.input_slice.stop > self.target_slice.stop:
                raise ValueError("Input slice stop must be before target slice stop.")

    def __len__(self) -> int:
        r"""Number of unique entries in the outer n-1 index levels."""
        return len(self.index)

    def __iter__(self) -> Iterator[PlainSample]:
        r"""Yield all the samples in the dataset."""
        for key in self.index:
            yield self[key]

    def __getitem__(self, key: Key, /) -> PlainSample:
        r"""Yield a single sample."""
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
