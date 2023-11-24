r"""Test `module.class`."""

import random
from collections.abc import Iterable, Iterator
from string import ascii_letters

from pytest import fixture, mark

from tsdm.random.samplers import HierarchicalSampler, RandomSampler


def exhaust_iterable(obj: Iterable, /) -> None:
    for _ in obj:
        pass


@mark.flaky(reruns=2)
def test_hierarchical_sampler() -> None:
    data = {
        "foo": [1, 2, 3, 4],
        "bar": [5, 6],
    }

    key_pairs = {(outer, inner) for outer in data for inner in data[outer]}

    subsamplers = {
        key: RandomSampler(value, shuffle=True) for key, value in data.items()
    }

    sampler = HierarchicalSampler(data, subsamplers=subsamplers, shuffle=True)

    for key in sampler:
        print(key)

    assert len(sampler) == len(key_pairs)
    assert set(sampler) == key_pairs
    assert list(sampler) != list(key_pairs)  # can fail with low probability


@fixture
def benchmark_data():
    # for each letter in the alphabet, list of random digits of size range(100, 1000)
    data = {
        letter: [random.randint(0, 9) for _ in range(random.randint(100, 1000))]
        for letter in ascii_letters
    }
    return data


@mark.benchmark
def test_benchmark_hierarchical_sampler(benchmark, benchmark_data):
    sampler = HierarchicalSampler(benchmark_data, shuffle=True)
    benchmark(exhaust_iterable, sampler)


@mark.benchmark
@mark.parametrize("method", ["iter_with_iter", "iter_with_yield"])
def test_iter_speed(benchmark, method):
    class Foo:
        def __init__(self):
            self.data = list(range(100))

        def __iter__(self) -> Iterator[int]:
            return iter(self.data)

    class Bar:
        def __init__(self):
            self.data = list(range(100))

        def __iter__(self) -> Iterator[int]:
            yield from self.data

    match method:
        case "iter_with_iter":
            foo = Foo()
            benchmark(exhaust_iterable, foo)
        case "iter_with_yield":
            bar = Bar()
            benchmark(exhaust_iterable, bar)
        case _:
            raise AssertionError
