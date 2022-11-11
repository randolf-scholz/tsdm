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


def test_time_series_dataset_task():
    """Test the TimeSeriesDatasetTask."""
    __logger__.info("Testing %s.", InSilicoTask)

    task = InSilicoTask()
    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.index, MultiIndex)
    assert isinstance(task.splits[0, "train"], TimeSeriesCollection)
    assert isinstance(task.samplers[0, "train"], HierarchicalSampler)
    assert isinstance(task.generators[0, "train"], InSilicoSampleGenerator)
    assert isinstance(task.dataloaders[0, "train"], DataLoader)
    assert isinstance(task.train_partition, dict)

    sampler = task.samplers[0, "train"]
    key = next(iter(sampler))
    sample = task.generators[0, "train"][key]
    assert isinstance(sample, tuple)

    dataloader = task.dataloaders[0, "train"]
    batch = next(iter(dataloader))
    assert batch is not None


def _main() -> None:
    test_time_series_dataset_task()


if __name__ == "__main__":
    _main()
