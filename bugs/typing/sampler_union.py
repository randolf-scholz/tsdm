#!/usr/bin/env python

import random
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Protocol,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    assert_type,
    overload,
    runtime_checkable,
)

import numpy as np
import pytest
from numpy.typing import NDArray

K = TypeVar("K")
V = TypeVar("V")
V_co = TypeVar("V_co", covariant=True)
Key_contra = TypeVar("Key_contra", contravariant=True)
Int_contra = TypeVar("Int_contra", contravariant=True, bound=int)


@runtime_checkable
class SupportsLenAndGetItem(Protocol[Key_contra, V_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, key: Key_contra, /) -> V_co: ...


@runtime_checkable
class MapDataset(Protocol[K, V_co]):  # indexed by keys.
    def __len__(self) -> int: ...
    def __getitem__(self, key: K, /) -> V_co: ...
    def keys(self) -> Iterable[K]: ...


@runtime_checkable
class IterableDataset(Protocol[V_co]):  # Assume indexed by 0...n-1
    def __len__(self) -> int: ...
    def __getitem__(self, key: int, /) -> V_co: ...
    def __iter__(self) -> Iterator[V_co]: ...


Dataset: TypeAlias = MapDataset[K, V] | IterableDataset[V]


@dataclass
class Sampler(Generic[K]):
    """Sample random indices that can be used to sample from a dataset."""

    data: Dataset[K, Any]
    # index: list[K]

    # if TYPE_CHECKING:
    #     @overload
    #     def __new__(cls, data_source: MapDataset[K, Any], /) -> "Sampler[K]": ...
    #     @overload
    #     def __new__(cls, data_source: IterableDataset, /) -> "Sampler[int]": ...
    #     def __new__(cls, data_source: Dataset[K, Any], /) -> "Sampler[K]":
    #         return super().__new__(cls)

    @overload
    def __init__(self: "Sampler[K]", data_source: MapDataset[K, Any], /) -> None: ...
    @overload
    def __init__(self: "Sampler[int]", data_source: IterableDataset, /) -> None: ...
    def __init__(self, data_source, /):
        self.data = data_source
        self.index: list[K]

        match data_source:
            case MapDataset() as map_data:  # in this case, K given by the Mapping
                self.index = list(map_data.keys())
            case IterableDataset() as seq_data:  # can we forcibly bind K to int?
                self.index = list(range(len(seq_data)))  # type: ignore[arg-type]
            case _:
                raise TypeError

    # def __post_init__(self) -> None:
    #     self.index: list[K]
    #
    #     match self.data:
    #         case MapDataset() as map_data:  # in this case, K given by the Mapping
    #             self.index = list(map_data.keys())
    #         case IterableDataset() as seq_data:  # can we forcibly bind K to int?
    #             self.index = list(range(len(seq_data)))  # type: ignore[arg-type]
    #         case _:
    #             raise TypeError

    def __iter__(self) -> Iterator[K]:
        random.shuffle(self.index)
        yield from self.index


def test_map_data_a() -> None:
    data: MapDataset[str, str] = {"x": "foo", "y": "bar"}
    sampler = Sampler(data)
    assert_type(sampler, Sampler[str])
    assert isinstance(sampler.index[0], str)


def test_map_data_b() -> None:
    data: MapDataset[int, str] = {10: "foo", 11: "bar"}
    sampler = Sampler(data)
    assert_type(sampler, Sampler[int])
    assert isinstance(sampler.index[0], int)


def test_map_data_c() -> None:
    data: MapDataset[str, int] = {"a": 10, "b": 11, "c": 12}
    sampler = Sampler(data)
    assert_type(sampler, Sampler[str])
    assert isinstance(sampler.index[0], str)


def test_raw_map_data() -> None:
    data = {10: "foo", 11: "bar"}
    sampler = Sampler(data)
    assert_type(sampler, Sampler[int])
    assert isinstance(sampler.index[0], int)


def test_seq_data() -> None:
    data: IterableDataset[str] = ["foo", "bar"]
    sampler = Sampler(data)
    assert_type(sampler, Sampler[int])  # Possibly Sampler[SupportsIndex]
    assert isinstance(sampler.index[0], int)


def test_raw_seq_data() -> None:
    data = ["foo", "bar"]
    sampler = Sampler(data)
    assert_type(sampler, Sampler[int])  # Possibly Sampler[SupportsIndex]
    assert isinstance(sampler.index[0], int)


def test_numpy_data() -> None:
    data: NDArray[np.int_] = np.array([1, 2, 3])
    sampler = Sampler(data)
    assert_type(sampler, Sampler[int])  # Possibly Sampler[SupportsIndex]
    assert isinstance(sampler.index[0], int)


if __name__ == "__main__":
    pytest.main()
