r"""Test the DampedPendulum task."""

import logging
from itertools import islice

from tsdm import tasks
from tsdm.random.samplers import HierarchicalSampler
from tsdm.timeseries.pandas import SplitTimeData
from tsdm.types.datasets import CallableDataset

__logger__ = logging.getLogger(__name__)


def test_damped_pendulum() -> None:
    r"""Test object."""
    __logger__.info("Testing %s", object)
    task = tasks.DampedPendulum_Ansari2023()

    train_indices = task.partitions[0, "train"]
    valid_indices = task.partitions[0, "valid"]
    test_indices = task.partitions[0, "test"]

    # check lengths
    assert len(train_indices) == 5000
    assert len(valid_indices) == 1000
    assert len(test_indices) == 1000

    # Match the reference's consecutive generation order, reused for every run.
    assert list(train_indices) == list(range(5000))
    assert list(valid_indices) == list(range(5000, 6000))
    assert list(test_indices) == list(range(6000, 7000))
    for fold in range(1, task.num_folds):
        assert task.split_masks[fold].equals(task.split_masks[0])

    # check pairwise disjointness
    assert set(train_indices).isdisjoint(valid_indices)
    assert set(train_indices).isdisjoint(test_indices)
    assert set(valid_indices).isdisjoint(test_indices)

    # test generator
    test_generator = task.generators[0, "test"]
    test_sampler = task.samplers[0, "test"]
    assert isinstance(test_generator, CallableDataset)
    assert isinstance(test_sampler, HierarchicalSampler)

    for key in islice(test_sampler, 10):
        sample = test_generator[key]
        assert isinstance(sample, SplitTimeData)
        assert sample.target_values is not None
        assert sample.context_times.index.equals(sample.context_values.index)
        assert sample.context_mask.index.equals(sample.context_values.index)
        assert sample.query_times.index.equals(sample.target_values.index)
        assert sample.query_mask.index.equals(sample.target_values.index)
        assert sample.context_mask.equals(sample.context_values.notna())
        assert sample.query_mask.equals(sample.target_values.notna())
