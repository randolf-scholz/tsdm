# ty: ignore[assert-type-unspellable-subtype]
r"""Test pprint decorators."""

from collections.abc import Iterator, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from typing import NamedTuple, Self, assert_type, overload

from tsdm.utils.pprint import (
    _pretty_dataclass,
    _pretty_mapping,
    _pretty_namedtuple,
    _pretty_sequence,
    _pretty_set,
    pretty_repr,
)

EXPECTED_SEQUENCE = """\
TestSequence<sequence>(
    0,
    1,
    2,
    3,
    4,
)"""
EXPECTED_NAMEDTUPLE = """\
TestNamedTuple<namedtuple>(
    a: 1,
    b: 'a',
    c: 1.25,
)"""
EXPECTED_DATACLASS = """\
TestDataclass<dataclass>(
    a: 1,
    b: 'a',
    c: 1.25,
)"""
EXPECTED_MAPPING = """\
TestMapping<mapping>(
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
)"""
EXPECTED_SET = """\
TestSet<set>(
    0,
    1,
    2,
    3,
    4,
)"""


def test_pprint_namedtuple() -> None:
    r"""Test pprint_namedtuple."""

    @_pretty_namedtuple
    class TestNamedTuple(NamedTuple):
        r"""Test NamedTuple."""

        a: int
        b: str
        c: float

    # ensure decorator works statically
    assert_type(TestNamedTuple, type[TestNamedTuple])

    # runtime check
    obj = TestNamedTuple(1, "a", 1.25)
    result = repr(obj)
    assert result == EXPECTED_NAMEDTUPLE


def test_pprint_repr_namedtuple() -> None:
    r"""Test pprint_namedtuple."""

    @pretty_repr
    class TestNamedTuple(NamedTuple):
        r"""Test NamedTuple."""

        a: int
        b: str
        c: float

    # ensure decorator works statically
    assert_type(TestNamedTuple, type[TestNamedTuple])

    # runtime check
    obj = TestNamedTuple(1, "a", 1.25)
    result = repr(obj)
    assert result == EXPECTED_NAMEDTUPLE


def test_pprint_dataclass() -> None:
    r"""Test pprint_dataclass."""

    @_pretty_dataclass
    @dataclass
    class TestDataclass:
        r"""Test Dataclass."""

        a: int
        b: str
        c: float

    # ensure decorator works statically
    assert_type(TestDataclass, type[TestDataclass])
    cls = _pretty_dataclass(TestDataclass)
    assert_type(cls, type[TestDataclass])

    # runtime check
    obj = TestDataclass(a=1, b="a", c=1.25)
    result = repr(obj)
    assert result == EXPECTED_DATACLASS


def test_pprint_repr_dataclass() -> None:
    r"""Test pprint_dataclass."""

    @pretty_repr
    @dataclass
    class TestDataclass:
        r"""Test Dataclass."""

        a: int
        b: str
        c: float

    # ensure decorator works statically
    assert_type(TestDataclass, type[TestDataclass])

    # runtime check
    obj = TestDataclass(1, "a", 1.25)
    result = repr(obj)
    assert result == EXPECTED_DATACLASS


def test_pprint_sequence() -> None:
    r"""Test pprint_sequence."""

    @_pretty_sequence
    class TestSequence(Sequence[int]):
        r"""Test Sequence."""

        items: list[int]

        def __init__(self, items: list[int]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        @overload
        def __getitem__(self, index: int) -> int: ...
        @overload
        def __getitem__(self, index: slice) -> Self: ...
        def __getitem__(self, index: int | slice) -> int | Self:
            if isinstance(index, slice):
                return self.__class__(self.items[index])
            return self.items[index]

    # ensure decorator works statically
    assert_type(TestSequence, type[TestSequence])

    # runtime check
    obj = TestSequence([0, 1, 2, 3, 4])
    result = repr(obj)
    assert result == EXPECTED_SEQUENCE


def test_pprint_repr_sequence() -> None:
    r"""Test pprint_sequence."""

    @pretty_repr
    class TestSequence(Sequence[int]):
        r"""Test Sequence."""

        items: list[int]

        def __init__(self, items: list[int]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        @overload
        def __getitem__(self, index: int) -> int: ...
        @overload
        def __getitem__(self, index: slice) -> Self: ...
        def __getitem__(self, index: int | slice) -> int | Self:
            if isinstance(index, slice):
                return self.__class__(self.items[index])
            return self.items[index]

    # ensure decorator works statically
    assert_type(TestSequence, type[TestSequence])

    # runtime check
    obj = TestSequence([0, 1, 2, 3, 4])
    result = repr(obj)
    assert result == EXPECTED_SEQUENCE


def test_pprint_mapping() -> None:
    r"""Test pprint_mapping."""

    @_pretty_mapping
    class TestMapping(Mapping[str, int]):
        r"""Test Mapping."""

        def __getitem__(self, key: str, /) -> int:
            return int(key)

        def __iter__(self) -> Iterator[str]:
            return iter(map(str, range(5)))

        def __len__(self) -> int:
            return 10

    # ensure decorator works statically
    assert_type(TestMapping, type[TestMapping])

    # runtime check
    obj = TestMapping()
    result = repr(obj)
    assert result == EXPECTED_MAPPING


def test_pprint_repr_mapping() -> None:
    r"""Test pprint_mapping."""

    @pretty_repr
    class TestMapping(Mapping[str, int]):
        r"""Test Mapping."""

        def __getitem__(self, key: str, /) -> int:
            return int(key)

        def __iter__(self) -> Iterator[str]:
            return iter(map(str, range(5)))

        def __len__(self) -> int:
            return 10

    # ensure decorator works statically
    assert_type(TestMapping, type[TestMapping])

    # runtime check
    obj = TestMapping()
    result = repr(obj)
    assert result == EXPECTED_MAPPING


def test_pprint_set() -> None:
    r"""Test pprint_set."""

    @_pretty_set
    class TestSet(AbstractSet[int]):
        r"""Test Set."""

        def __contains__(self, item: object) -> bool:
            return item in range(5)

        def __iter__(self) -> Iterator[int]:
            return iter(range(5))

        def __len__(self) -> int:
            return 5

    # ensure decorator works statically
    assert_type(TestSet, type[TestSet])
    # runtime check
    obj = TestSet()
    result = repr(obj)
    assert result == EXPECTED_SET


def test_pprint_repr_set() -> None:
    r"""Test pprint_set."""

    @pretty_repr
    class TestSet(AbstractSet[int]):
        r"""Test Set."""

        def __contains__(self, item: object) -> bool:
            return item in range(5)

        def __iter__(self) -> Iterator[int]:
            return iter(range(5))

        def __len__(self) -> int:
            return 5

    # ensure decorator works statically
    assert_type(TestSet, type[TestSet])
    # runtime check
    obj = TestSet()
    result = repr(obj)
    assert result == EXPECTED_SET


def test_pprint_sequence_static() -> None:
    class Foo(Sequence): ...

    @_pretty_sequence
    class Bar(Sequence): ...

    @_pretty_sequence(linebreaks=True)
    class Baz(Sequence): ...

    FooBar = _pretty_sequence(Foo)
    FooBaz = _pretty_sequence(linebreaks=True)(Foo)

    # self-consistency
    assert_type(FooBar, type[Foo])
    assert_type(FooBaz, type[Foo])
    assert_type(Bar, type[Bar])
    assert_type(Baz, type[Baz])
    assert type(FooBar) is type(Foo)
    assert type(FooBaz) is type(Foo)


def test_pprint_mapping_static() -> None:
    class Foo(Mapping): ...

    @_pretty_mapping
    class Bar(Mapping): ...

    @_pretty_mapping(linebreaks=True)
    class Baz(Mapping): ...

    FooBar = _pretty_mapping(Foo)
    FooBaz = _pretty_mapping(linebreaks=True)(Foo)
    # self-consistency
    assert_type(FooBar, type[Foo])
    assert_type(FooBaz, type[Foo])
    assert_type(Bar, type[Bar])
    assert_type(Baz, type[Baz])
    assert type(FooBar) is type(Foo)
    assert type(FooBaz) is type(Foo)


def test_pprint_set_static() -> None:
    class Foo(AbstractSet): ...

    @_pretty_set
    class Bar(AbstractSet): ...

    @_pretty_set(linebreaks=True)
    class Baz(AbstractSet): ...

    FooBar = _pretty_set(Foo)
    FooBaz = _pretty_set(linebreaks=True)(Foo)
    # self-consistency
    assert_type(FooBar, type[Foo])
    assert_type(FooBaz, type[Foo])
    assert_type(Bar, type[Bar])
    assert_type(Baz, type[Baz])
    assert type(FooBar) is type(Foo)
    assert type(FooBaz) is type(Foo)
