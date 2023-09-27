#!/usr/bin/env python
r"""Test the DampedPendulum task."""

import logging

from pandas import DataFrame

from tsdm import tasks

logging.basicConfig(level=logging.INFO)
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
    for idx in test_indices[:20]:
        assert isinstance(test_generator[idx], DataFrame)


def _main() -> None:
    test_damped_pendulum()


if __name__ == "__main__":
    _main()
