r"""Pandas-based utilities for time series prediction."""

__all__ = ["SplitTimeData", "make_sample", "make_sample_factory"]

import math
import warnings
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, replace
from functools import partial
from typing import Any, Optional, Self

from pandas import DataFrame, Index, Series

from tsdm.datatools import CallableDataset
from tsdm.pprint import pprint_repr
from tsdm.timeseries import abstract

from .datasets import PandasTS, PandasTSC


@pprint_repr
@dataclass(slots=True, frozen=True)
class SplitTimeData(abstract.SplitTimeData[Series | DataFrame]):
    r"""A pandas-backed prediction sample with separate context and query times."""

    _: KW_ONLY

    context_times: Series
    r"""Timestamps associated with the context values."""
    context_values: DataFrame
    r"""Values available to the model as prediction context."""
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


def make_sample(
    dataset: PandasTS | PandasTSC,
    key: Any,
    /,
    *,
    targets: Iterable[str],
    observables: Iterable[str],
    covariates: Iterable[str],
    sparse_index: bool = False,
    sparse_columns: bool = False,
) -> SplitTimeData:
    r"""Create a split-format prediction sample from pandas time-series data.

    Observables are available only in the observation horizon, covariates are
    available throughout both horizons, and targets are available only in the
    prediction horizon. The two horizons may overlap, which also supports
    imputation and reconstruction tasks.

    Args:
        dataset: A single time series or a collection of time series.
        key: A pair of observation and prediction horizons for a single series,
            or a pair of a series key and horizons for a collection.
        targets: Columns to expose as prediction targets.
        observables: Columns to expose in the observation horizon.
        covariates: Columns to expose throughout both horizons.
        sparse_index: Whether to remove rows that contain no context or target values.
        sparse_columns: Whether to omit unused columns from the returned frames.

    Returns:
        A sample with separate context and query time axes.

    Note:
        This low-level function does not validate the selected columns. Use
        make_sample_factory when creating many samples from one dataset.
    """
    match dataset:
        case PandasTS() as tsd:
            # Single-series keys contain only the pair of horizons.
            horizons = key
            static_covariates = tsd.static_covariates
        case PandasTSC() as tsc:
            # Collection keys additionally identify the selected time series.
            try:
                outer_key, horizons = key
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid key: {key!r}") from exc
            tsd = tsc[outer_key]
            static_covariates = (
                None
                if tsc.static_covariates is None
                else tsc.static_covariates.loc[outer_key]
            )
        case _:
            raise TypeError(f"Invalid dataset type: {type(dataset)=}")

    try:
        observation_horizon, prediction_horizon = horizons
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid key: {key!r}") from exc

    observable_columns = Index(observables)
    target_columns = Index(targets)
    covariate_columns = Index(covariates)

    ts_observed: DataFrame = tsd.timeseries.loc[observation_horizon]
    ts_predicted: DataFrame = tsd.timeseries.loc[prediction_horizon]
    # Keep a shared timestamp only once when the two horizons overlap.
    joint_horizon_index = ts_observed.index.union(ts_predicted.index)

    joint_horizon_mask = tsd.timeseries.index.isin(joint_horizon_index)
    ts = tsd.timeseries.loc[joint_horizon_mask]
    ts_observed_mask = ts.index.isin(ts_observed.index)
    ts_predicted_mask = ts.index.isin(ts_predicted.index)

    context_columns = (
        ts.columns.intersection(observable_columns.union(covariate_columns), sort=False)
        if sparse_columns
        else ts.columns
    )
    # Dense samples retain the complete schema; sparse samples retain only
    # columns that can carry context or target values.
    prediction_columns = target_columns if sparse_columns else ts.columns
    context_values = DataFrame(math.nan, index=ts.index, columns=context_columns)
    target_values = DataFrame(
        math.nan,
        index=ts.index[ts_predicted_mask],
        columns=prediction_columns,
    )

    # Values begin as missing. These assignments encode which channels are
    # available in each horizon, while the masks are derived below.
    context_values.loc[ts_observed_mask, observable_columns] = ts.loc[
        ts_observed_mask, observable_columns
    ]
    context_values.loc[:, covariate_columns] = ts.loc[:, covariate_columns]
    target_values.loc[:, target_columns] = ts.loc[ts_predicted_mask, target_columns]

    sample = SplitTimeData(
        context_times=context_values.index.to_series().copy(),
        context_values=context_values,
        context_mask=context_values.notna(),
        query_times=target_values.index.to_series().copy(),
        query_mask=target_values.notna(),
        target_values=target_values,
        static_covariates=static_covariates,
    )
    # Row sparsification is deliberately last so all masks stay aligned.
    return sample.drop_null_rows() if sparse_index else sample


def make_sample_factory(
    dataset: PandasTS | PandasTSC,
    /,
    *,
    targets: Iterable[str],
    observables: Iterable[str],
    covariates: Iterable[str],
    sparse_index: bool = False,
    sparse_columns: bool = False,
) -> CallableDataset[Any, SplitTimeData]:
    r"""Create a validated factory for pandas split-format prediction samples.

    The selected columns are normalized and validated once when the factory is
    created. The returned map-style dataset accepts the same sample key as
    make_sample and is suitable for use with a sampler or DataLoader.

    Args:
        dataset: A single time series or a collection of time series.
        targets: Columns to expose as prediction targets.
        observables: Columns to expose in the observation horizon.
        covariates: Columns to expose throughout both horizons.
        sparse_index: Whether generated samples omit empty rows.
        sparse_columns: Whether generated samples omit unused columns.

    Returns:
        A dataset that maps sample keys to validated split-format samples.
    """
    targets = Index(targets)
    observables = Index(observables)
    covariates = Index(covariates)
    _validate_columns(dataset, targets, observables, covariates)
    return CallableDataset(
        partial(
            make_sample,
            dataset,
            targets=targets,
            observables=observables,
            covariates=covariates,
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
        )
    )


def _validate_columns(
    dataset: PandasTS | PandasTSC,
    targets: Index,
    observables: Index,
    covariates: Index,
    /,
) -> None:
    r"""Validate selected columns against a pandas time-series dataset."""
    ts_columns = set(dataset.timeseries.columns)
    target_columns = set(targets)
    observable_columns = set(observables)
    covariate_columns = set(covariates)

    if columns := covariate_columns - ts_columns:
        raise ValueError(f"Covariates {columns} not in found timeseries columns!")
    if columns := observable_columns - ts_columns:
        raise ValueError(f"Observables {columns} not in found timeseries columns!")
    if columns := target_columns - ts_columns:
        raise ValueError(f"Targets {columns} not found in timeseries columns!")
    if columns := covariate_columns & observable_columns:
        raise ValueError(f"Covariates and observables not disjoint! {columns}.")
    if columns := ts_columns - (
        observable_columns | target_columns | covariate_columns
    ):
        warnings.warn(f"Unused columns in timeseries: {columns}", stacklevel=3)
