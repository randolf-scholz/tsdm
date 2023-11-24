"""Test the RandomSampler."""

from typing import assert_type

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pytest import mark

from tsdm.data import IndexableDataset, MapDataset
from tsdm.random.samplers import RandomSampler


@mark.flaky(reruns=2)  # 1 in 10¹² chance of failure
def test_random_sampler_dict() -> None:
    r"""Test RandomSampler."""
    data: dict[int, object] = {
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
    assert_type(sampler, RandomSampler[object])

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
    data: MapDataset[str, int] = {"x": 0, "y": 1}
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
    data: IndexableDataset[str] = ["foo", "bar"]
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[str])
    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, str)


def test_seq_data_no_hint() -> None:
    data = ["foo", "bar"]
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[str])
    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, str)


def test_numpy_data() -> None:
    data: NDArray[np.str_] = np.array(["1", "2", "3"], dtype=np.str_)
    sampler = RandomSampler(data)
    assert_type(sampler, RandomSampler[np.str_])  # type: ignore[assert-type]
    # check that we can iterate over the index
    for val in sampler:
        assert isinstance(val, np.str_)


PYTHON_STRINGS = ["foo", "bar", "baz", "qux", "quux", "quuz", "corge"]
MAPPED_STRINGS = {2 * k + 1: s for k, s in enumerate(PYTHON_STRINGS)}  # generic index
STRING_DATA = {
    "list": PYTHON_STRINGS,
    "tuple": tuple(PYTHON_STRINGS),
    "dict": MAPPED_STRINGS,
    "numpy": np.array(PYTHON_STRINGS, dtype=np.str_),
    "index": pd.Index(PYTHON_STRINGS, dtype="string"),
    "series": pd.Series(MAPPED_STRINGS, dtype="string"),
    "series-pyarrow": pd.Series(MAPPED_STRINGS, dtype="string[pyarrow]"),
}


@mark.parametrize("data", STRING_DATA.values(), ids=STRING_DATA)
def test_string_data(data):
    sampler = RandomSampler(data, shuffle=True)

    assert len(sampler) == len(PYTHON_STRINGS)
    assert set(sampler) == set(PYTHON_STRINGS)
    assert list(sampler) != list(PYTHON_STRINGS)
