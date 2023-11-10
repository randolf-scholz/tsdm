"""Test the RandomSampler."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray
from pytest import mark

from tsdm.random.samplers import RandomSampler
from tsdm.utils.data import IterableDataset, MapDataset


@mark.flaky(reruns=2)  # 1 in 10¹² chance of failure
def test_random_sampler_dict() -> None:
    r"""Test RandomSampler."""
    data = {
        0: 0,
        1: 1,
        2: 2,
        3: "a",
        4: "b",
        5: "c",
        6: None,
        7: int,
        8: float,
        9: Ellipsis,
    }

    sampler = RandomSampler(data, shuffle=True)
    assert_type(sampler, RandomSampler[int])

    # check length
    assert len(sampler) == len(data) == 10

    # check that the elements are the same
    assert set(sampler) == set(data.values())

    # check that the order is random (might fail with probability 1/n!)
    assert list(sampler) != list(data.values())


@mark.flaky(reruns=2)  # 1 in 10¹² chance of failure
def test_random_sampler_list() -> None:
    r"""Test RandomSampler."""
    indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    sampler = RandomSampler(indices, shuffle=True)
    assert_type(sampler, RandomSampler[int])

    # check length
    assert len(sampler) == len(indices) == 10

    # check that the elements are the same
    assert set(sampler) == set(indices)

    # check that the order is random (might fail with probability 1/n!)
    assert list(sampler) != list(indices)


def test_map_data_a() -> None:
    data: MapDataset[str, str] = {"x": 0, "y": 1}
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[int])

    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, int)


def test_map_data_b() -> None:
    data: MapDataset[int, str] = {10: "foo", 11: "bar"}
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[str])

    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, str)


def test_map_data_c() -> None:
    data: MapDataset[str, int] = {"a": 10, "b": 11, "c": 12}
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[int])

    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, int)


def test_map_data_no_typehint() -> None:
    data = {10: "foo", 11: "bar"}
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[str])

    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, str)


def test_seq_data() -> None:
    data: IterableDataset[str] = ["foo", "bar"]
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[str])  # Possibly RandomSampler[SupportsIndex]

    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, str)


def test_seq_data_no_hint() -> None:
    data = ["foo", "bar"]
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[str])  # Possibly RandomSampler[SupportsIndex]

    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, str)


def test_numpy_data() -> None:
    data: NDArray[np.int_] = np.array(["1", "2", "3"], dtype=np.str_)
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[np.str_])

    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, np.str_)
