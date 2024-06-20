# fmt: off
from typing import Any, Iterator, Protocol, Reversible, TypeAlias, TypeVar, assert_type

K = TypeVar("K")
V_co = TypeVar("V_co", covariant=True)

class IndexableDataset(Protocol[V_co]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[V_co]: ...
    def __getitem__(self, index: int, /) -> V_co: ...

class MapDataset(Protocol[K, V_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, key: K, /) -> V_co: ...
    def keys(self) -> Reversible[K] | IndexableDataset[K]: ...

Dataset: TypeAlias = IndexableDataset[V_co] | MapDataset[Any, V_co]

def as_indexable(obj: IndexableDataset[K]) -> IndexableDataset[K]: return obj
def as_map(obj: MapDataset[K, V_co]) -> MapDataset[K, V_co]: return obj
def as_dataset(obj: Dataset[V_co]) -> Dataset[V_co]: return obj

data: dict[int, object] = {0: 1, 1: None, 3: type}
assert_type(as_indexable(data), IndexableDataset[object])
assert_type(as_map(data), MapDataset[int, object])
assert_type(as_dataset(data), Dataset[object])  # expected "IndexableDataset[Never] | MapDataset[Any, Never]"

from typing import Protocol, Self, overload

T_co = TypeVar("T_co", covariant=True)


class SupportsSlicing(Protocol[T_co]):
    @overload
    def __getitem__(self, index: int, /) -> T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...

x: P = object()
