"""Task-specific utilities."""

__all__ = ["PaddedBatch", "Inputs", "Sample", "collate_timeseries"]

from math import nan as NAN
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tsdm.utils.strings import repr_namedtuple


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self)


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self)


class PaddedBatch(NamedTuple):
    r"""A single sample of the data."""

    # N = number of observations + K + padding
    t: Tensor  # B×N:   the padded timestamps/queries.
    x: Tensor  # B×N×D: the padded input values.
    y: Tensor  # B×K×F: the padded target values.

    mq: Tensor  # B×N: the queries mask.
    mx: Tensor  # B×N: the inputs  mask.
    my: Tensor  # B×K: the targets mask.

    def __repr__(self) -> str:
        return repr_namedtuple(self)


# @torch.jit.script  # seems to break things
def collate_timeseries(batch: list[Sample]) -> PaddedBatch:
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

        # get whole time interval
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
