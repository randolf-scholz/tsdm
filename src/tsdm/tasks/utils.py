"""Task-specific utilities."""

__all__ = [
    # NamedTuples
    "PaddedBatch",
    "Inputs",
    "Sample",
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
from typing import Any, Generic, NamedTuple

import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset

from tsdm.types.variables import any_co as T_co
from tsdm.utils.strings import pprint_repr


@pprint_repr
class Inputs(NamedTuple, Generic[T_co]):
    r"""A single sample of the data."""

    t: T_co
    x: T_co
    t_target: T_co


@pprint_repr
class Sample(NamedTuple, Generic[T_co]):
    r"""A single sample of the data."""

    key: Hashable
    inputs: Inputs[T_co]
    targets: T_co
    original: Any = None


@pprint_repr
class PaddedBatch(NamedTuple):
    r"""A single sample of the data."""

    # N = number of observations + K + padding
    t: Tensor  # B×N:   the padded timestamps/queries.
    x: Tensor  # B×N×D: the padded input values.
    y: Tensor  # B×K×F: the padded target values.

    mq: Tensor  # B×N: the 'queries' mask.
    mx: Tensor  # B×N: the 'inputs'  mask.
    my: Tensor  # B×K: the 'targets' mask.


# @torch.jit.script  # seems to break things
def collate_timeseries(batch: list[Sample[Tensor]]) -> PaddedBatch:
    r"""Collate timeseries into padded batch.

    Assumptions:
        - t_target is sorted.

    Transform the data slightly: `t, x, t_target → T, X where X[t_target:] = NAN`.
    """
    masks_inputs: list[Tensor] = []
    masks_queries: list[Tensor] = []
    masks_target: list[Tensor] = []
    padded_inputs: list[Tensor] = []
    padded_queries: list[Tensor] = []
    padded_targets: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get the whole time interval
        t_combined = torch.cat((t, t_target))
        sorted_idx = torch.argsort(t_combined)

        # pad the x-values
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]),
            fill_value=NAN,
            device=x.device,
            dtype=x.dtype,
        )
        x_padded = torch.cat((x, x_padding))

        # create a mask for looking up the target values
        m_targets = torch.ones_like(t_target, dtype=torch.bool)
        m_queries = torch.cat(
            [
                torch.zeros_like(t, dtype=torch.bool),
                m_targets,
            ]
        )
        m_inputs = ~m_queries

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
        self.index: pd.Index = self.data_source.reset_index(level=-1).index.unique()
        self.columns: pd.Index = self.data_source.columns

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
        self.observables = pd.Index(self.observables)
        self.targets = pd.Index(self.targets)
        self.covariates = pd.Index(self.covariates)

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

        # create the sample
        inputs = Inputs(
            t=inputs.index.copy(),
            x=inputs.copy(),
            t_target=targets.index.copy(),
        )
        sample = Sample(key=key, inputs=inputs, targets=targets, original=ts)
        return sample
