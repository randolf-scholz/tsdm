#!/usr/bin/env python
r"""Test task implementation with InSilicoData."""

import logging

from pandas import DataFrame, MultiIndex
from torch.utils.data import DataLoader

from tsdm.datasets import TimeSeriesCollection
from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import InSilicoSampleGenerator, InSilicoTask


def test_insilico_task(SplitID=(0, "train")):
    r"""Test the TimeSeriesDatasetTask."""
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
    assert isinstance(key, tuple)
    fold = task.generators[SplitID]
    print("key", type(key), key)
    print("fold", type(fold), fold)
    sample = fold[key]
    assert isinstance(sample, tuple)

    dataloader = task.dataloaders[SplitID]
    batch = next(iter(dataloader))
    assert batch is not None


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_insilico_task()


if __name__ == "__main__":
    _main()
