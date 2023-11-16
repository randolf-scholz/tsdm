"""Task-specific utilities."""

__all__ = [
    # NamedTuples
    "PaddedBatch",
    "Sample",
    "TimeSeriesSample",
    # Functions
    "collate_timeseries",
    # Classes
    "FixedSliceSampleGenerator",
]

import warnings
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import KW_ONLY, dataclass
from math import nan as NAN
from types import NotImplementedType

import torch
from pandas import DataFrame, Index
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import Any, NamedTuple, Optional

from tsdm.utils.strings import pprint_repr


@pprint_repr
class Sample(NamedTuple):
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


# @torch.jit.script  # seems to break things
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


@pprint_repr
@dataclass
class FixedSliceSampleGenerator(TorchDataset[Sample]):
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

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over all the samples in the dataset."""
        for key in self.index:
            yield self[key]

    def __getitem__(self, key: Hashable) -> Sample:
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

        return Sample(
            key=key,
            t=inputs.index.copy(),
            x=inputs.copy(),
            t_target=targets.index.copy(),
            targets=targets,
        )
