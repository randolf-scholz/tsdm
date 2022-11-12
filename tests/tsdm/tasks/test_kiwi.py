#!/usr/bin/env python
r"""Test task implementation with InSilicoData."""

import logging

import pytest
from pandas import DataFrame, MultiIndex
from torch.utils.data import DataLoader

from tsdm.datasets import TimeSeriesCollection
from tsdm.random.samplers import HierarchicalSampler
from tsdm.tasks import KiwiSampleGenerator, KiwiTask

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@pytest.mark.slow
def test_kiwi_task():
    r"""Test the KiwiTask."""
    LOGGER = __logger__.getChild(KiwiTask.__name__)
    LOGGER.info("Testing.")

    task = KiwiTask()
    assert isinstance(task.folds, DataFrame)
    assert isinstance(task.index, MultiIndex)
    assert isinstance(task.splits[0, "train"], TimeSeriesCollection)
    assert isinstance(task.samplers[0, "train"], HierarchicalSampler)
    assert isinstance(task.generators[0, "train"], KiwiSampleGenerator)
    assert isinstance(task.dataloaders[0, "train"], DataLoader)
    assert isinstance(task.train_partition, dict)

    sampler = task.samplers[0, "train"]
    key = next(iter(sampler))
    sample = task.generators[0, "train"][key]
    assert isinstance(sample, tuple)

    dataloader = task.dataloaders[0, "train"]
    batch = next(iter(dataloader))
    assert isinstance(batch, list)
    assert isinstance(batch[0], tuple)


def _main() -> None:
    test_kiwi_task()


if __name__ == "__main__":
    _main()
