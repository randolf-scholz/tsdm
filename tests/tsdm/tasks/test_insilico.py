#!/usr/bin/env python
r"""Test task implementation with InSilicoData."""

import logging

from pandas import DataFrame, MultiIndex
from torch.utils.data import DataLoader

from tsdm.datasets import TimeSeriesCollection
from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import InSilicoSampleGenerator, InSilicoTask

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


def test_insilico_task(SplitID=(0, "train")):
    r"""Test the TimeSeriesDatasetTask."""
    LOGGER = __logger__.getChild(InSilicoTask.__name__)
    LOGGER.info("Testing.")

    task = InSilicoTask()
    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.index, MultiIndex)
    assert isinstance(task.splits[SplitID], TimeSeriesCollection)
    assert isinstance(task.samplers[SplitID], HierarchicalSampler)
    assert isinstance(task.generators[SplitID], InSilicoSampleGenerator)
    assert isinstance(task.dataloaders[SplitID], DataLoader)
    assert task.collate_fns[SplitID] is NotImplemented
    assert isinstance(task.train_split, dict)

    sampler = task.samplers[SplitID]
    key = next(iter(sampler))
    sample = task.generators[SplitID][key]
    assert isinstance(sample, tuple)

    dataloader = task.dataloaders[SplitID]
    batch = next(iter(dataloader))
    assert batch is not None


def _main() -> None:
    test_insilico_task()


if __name__ == "__main__":
    _main()
