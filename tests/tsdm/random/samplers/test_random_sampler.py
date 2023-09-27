#!/usr/bin/env python
"""Test the RandomSampler."""

from pandas import Index
from pytest import mark

from tsdm.random.samplers import RandomSampler


@mark.flaky(reruns=2)  # 1 in 10¹² chance of failure
def test_random_sampler() -> None:
    r"""Test RandomSampler."""
    map_style_dataset = {
        0: 0,
        1: 1,
        2: 2,
        "a": 3,
        "b": 4,
        "c": 5,
        None: 6,
        int: 7,
        float: 8,
        Ellipsis: 9,
    }
    sampler = RandomSampler(map_style_dataset)

    # check length
    assert len(sampler) == len(map_style_dataset) == 10

    # check that the elements are the same
    assert set(sampler) == set(map_style_dataset.values())

    # check that the order is random (might fail with probability 1/n!)
    assert list(sampler) != list(map_style_dataset.values())


@mark.flaky(reruns=2)  # 1 in 10¹² chance of failure
def test_random_sampler_index() -> None:
    r"""Test RandomSampler."""
    indices = Index([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    sampler = RandomSampler(indices)

    # check length
    assert len(sampler) == len(indices) == 10

    # check that the elements are the same
    assert set(sampler) == set(indices)

    # check that the order is random (might fail with probability 1/n!)
    assert list(sampler) != list(indices)


def _main() -> None:
    test_random_sampler()


if __name__ == "__main__":
    _main()
