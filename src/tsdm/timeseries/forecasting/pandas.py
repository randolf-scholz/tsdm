r"""Pandas-based utilities for time series forecasting."""

__all__ = ["SplitTimeData", "PandasForecastingDataset"]

import math
import warnings
from dataclasses import KW_ONLY, dataclass, replace
from typing import Any, Optional, Self, cast

from pandas import DataFrame, Index, Series

from tsdm.constants import UNDEFINED
from tsdm.pprint import pprint_repr
from tsdm.timeseries import PandasTS, PandasTSC
from tsdm.types import SupportsGetItem

from . import abstract


@pprint_repr
@dataclass(slots=True, frozen=True)
class SplitTimeData(abstract.SplitTimeData[Series | DataFrame]):
    r"""A pandas-backed forecasting sample with separate context and query times."""

    _: KW_ONLY

    context_times: Series
    r"""Timestamps associated with the context values."""
    context_values: DataFrame
    r"""Values available to the model as forecasting context."""
    context_mask: DataFrame
    r"""Mask indicating which context values are available."""

    query_times: Series
    r"""Timestamps at which predictions are requested."""
    query_mask: DataFrame
    r"""Mask indicating which target values should be predicted."""
    target_values: Optional[DataFrame] = None  # pyright: ignore[reportIncompatibleMethodOverride]
    r"""Target values at the query times."""

    static_covariates: Optional[DataFrame] = None  # pyright: ignore[reportIncompatibleMethodOverride]
    r"""Static covariates associated with the time series."""

    def drop_null_rows(self) -> Self:
        r"""Drop rows that contain only NAN values."""
        context_values = self.context_values.dropna(how="all")

        if self.target_values is None:
            return replace(
                self,
                context_times=self.context_times.loc[context_values.index],
                context_values=context_values,
                context_mask=self.context_mask.loc[context_values.index],
            )

        target_values = self.target_values.dropna(how="all")
        return replace(
            self,
            context_times=self.context_times.loc[context_values.index],
            context_values=context_values,
            context_mask=self.context_mask.loc[context_values.index],
            query_times=self.query_times.loc[target_values.index],
            query_mask=self.query_mask.loc[target_values.index],
            target_values=target_values,
        )


@pprint_repr
@dataclass(slots=True)
class PandasForecastingDataset[KeyT](SupportsGetItem[KeyT, SplitTimeData]):
    r"""Creates sample from a TimeSeriesCollection.

    This class is responsible for creating samples from a TimeSeriesCollection.
    It acts as a Map-Stype Dataset, keys should be created from an appriopriate Sampler instance.

    There are different modus operandi for creating samples from a TimeSeriesCollection.

    Note:
        - column-sparse
        - separate context and query times
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
        - time series classification task: empty forecasting horizon.
        - time series imputation task: observation horizon and forecasting horizon overlap
        - time series forecasting task: observation horizon and forecasting horizon
        - time series forecasting task (autoregressive): observables = targets
        - time series event forecasting: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} given n
        - time series event forecasting++: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} and n

    Notes:
        Samples are frozen dataclasses that structurally implement
        :class:`SeparateTimeSample`.
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
    sparse_index: bool = False
    r"""Whether to drop sparse rows from the index."""
    sparse_columns: bool = False
    r"""Whether to drop sparse cols from the data."""

    def __post_init__(self) -> None:
        r"""Post init."""
        self.targets = Index([]) if self.targets is UNDEFINED else Index(self.targets)
        self.observables = (
            self.dataset.timeseries.columns.copy()
            if self.observables is UNDEFINED
            else Index(self.observables)
        )
        self.covariates = (
            Index([]) if self.covariates is UNDEFINED else Index(self.covariates)
        )
        self.validate()

    def __getitem__(self, key: KeyT, /) -> SplitTimeData:
        return self.make_sample(
            key,
            sparse_index=self.sparse_index,
            sparse_columns=self.sparse_columns,
        )

    def make_sample(
        self,
        key: KeyT,
        *,
        sparse_index: bool = False,
        sparse_columns: bool = False,
    ) -> SplitTimeData:
        r"""Create a sample from a TimeSeriesCollection.

        Args:
            key: The key of the sample - e.g. tuple[outer_index, (obs_rane, forecasting_range)].
            sparse_index: Whether to drop rows that contain only NAN values.
            sparse_columns: Whether to drop columns that contain only NAN values.
        """
        match self.dataset:
            case PandasTS() as tsd:
                horizons = cast("Any", key)
                static_covariates = tsd.static_covariates
            case PandasTSC() as tsc:
                try:
                    outer_key, horizons = cast("Any", key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid key: {key!r}") from exc
                tsd = tsc[outer_key]
                static_covariates = (
                    None
                    if tsc.static_covariates is None
                    else tsc.static_covariates.loc[outer_key]
                )
            case _:
                raise TypeError(f"Invalid dataset type: {type(self.dataset)=}")

        try:
            observation_horizon, forecasting_horizon = horizons
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid key: {key!r}") from exc

        observables = Index(self.observables)
        targets = Index(self.targets)
        covariates = Index(self.covariates)

        # extract the relevant time series and their joint index
        ts_observed: DataFrame = tsd.timeseries.loc[observation_horizon]
        ts_forecast: DataFrame = tsd.timeseries.loc[forecasting_horizon]
        joint_horizon_index = ts_observed.index.union(ts_forecast.index)

        joint_horizon_mask = tsd.timeseries.index.isin(joint_horizon_index)
        ts = tsd.timeseries.loc[joint_horizon_mask]
        ts_observed_mask = ts.index.isin(ts_observed.index)
        ts_forecast_mask = ts.index.isin(ts_forecast.index)

        context_columns = (
            ts.columns.intersection(observables.union(covariates), sort=False)
            if sparse_columns
            else ts.columns
        )
        target_columns = targets if sparse_columns else ts.columns
        context_values = DataFrame(math.nan, index=ts.index, columns=context_columns)
        target_values = DataFrame(
            math.nan,
            index=ts.index[ts_forecast_mask],
            columns=target_columns,
        )

        # Observables are available in the observation horizon. Time-varying
        # covariates are available throughout the joint horizon.
        context_values.loc[ts_observed_mask, observables] = ts.loc[
            ts_observed_mask, observables
        ]
        context_values.loc[:, covariates] = ts.loc[:, covariates]

        # Targets are available in the forecasting horizon. Assigning from the
        # unmasked data preserves values where observation and forecast overlap.
        target_values.loc[:, targets] = ts.loc[ts_forecast_mask, targets]

        context_times = context_values.index.to_series().copy()
        context_mask = context_values.notna()
        query_times = target_values.index.to_series().copy()
        query_mask = target_values.notna()

        # assemble sample
        sample = SplitTimeData(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            target_values=target_values,
            static_covariates=static_covariates,
        )

        if sparse_index:
            sample = sample.drop_null_rows()

        return sample

    def validate(self) -> None:
        r"""Validate that chosen columns are present."""
        ts = self.dataset.timeseries
        observables = set(self.observables)
        targets = set(self.targets)
        covariates = set() if self.covariates is None else set(self.covariates)
        ts_columns = set(ts.columns)

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
