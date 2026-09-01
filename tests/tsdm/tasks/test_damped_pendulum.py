r"""Test the DampedPendulum task."""

import logging
from itertools import islice

from tsdm import tasks
from tsdm.datatools import CallableDataset
from tsdm.random.samplers import HierarchicalSampler
from tsdm.timeseries.pandas import SplitTimeData

__logger__ = logging.getLogger(__name__)


def test_damped_pendulum() -> None:
    r"""Test object."""
    __logger__.info("Testing %s", object)
    task = tasks.DampedPendulum_Ansari2023()

    train_indices = task.splits[0, "train"]
    valid_indices = task.splits[0, "valid"]
    test_indices = task.splits[0, "test"]

    # check lengths
    assert len(train_indices) == 5000
    assert len(valid_indices) == 1000
    assert len(test_indices) == 1000

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
