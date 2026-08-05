r"""Utilities for time series data handling."""

__all__ = [
    # NamedTuples
    "TimeSeriesSample",
    "PaddedBatch",
    # functions
    "collate_timeseries",
]

from math import nan as NAN
from typing import NamedTuple, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tsdm.pprint import pprint_repr


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
        m_queries = torch.cat(
            [
                torch.zeros_like(t_inputs, dtype=torch.bool),
                torch.ones_like(t_target, dtype=torch.bool),
            ]
        )
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
