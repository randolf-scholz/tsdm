r"""Test which builtins satisfy which protocols."""

import pprint
from collections import abc, defaultdict

import pytest

ABCS: list[type] = [
    abc.Container,
    abc.Hashable,
    abc.Iterable,
    abc.Iterator,
    abc.Generator,
    abc.Reversible,
    abc.Sized,
    abc.Callable,  # type: ignore[list-item]  # pyright: ignore[reportAssignmentType]
    abc.Collection,
    abc.Sequence,
    abc.MutableSequence,
    abc.Set,
    abc.MutableSet,
    abc.Mapping,
    abc.MutableMapping,
    abc.MappingView,
    abc.KeysView,
    abc.ItemsView,
    abc.ValuesView,
]


CLASSES: list[type] = [
    set,
    dict,
    frozenset,
    list,
    tuple,
    str,
    bytes,
    bytearray,
    memoryview,
    range,
    slice,
    type,
    object,
    complex,
    float,
    int,
    bool,
    map,
    filter,
    reversed,
]

BUILTINS: dict[type, list[type]] = {
    abc.Callable: [type],
    abc.Collection: [set, dict, frozenset, list, tuple, str, bytes, bytearray, memoryview, range],
    abc.Container: [set, dict, frozenset, list, tuple, str, bytes, bytearray, memoryview, range],
    abc.Iterable: [set, dict, frozenset, list, tuple, str, bytes, bytearray, memoryview, range, map, filter, reversed],
    abc.Iterator: [map, filter, reversed],
    abc.Mapping: [dict],
    abc.MutableMapping: [dict],
    abc.MutableSequence: [list, bytearray],
    abc.MutableSet: [set],
    abc.Reversible: [dict, list, tuple, str, bytes, bytearray, memoryview, range],
    abc.Sequence: [list, tuple, str, bytes, bytearray, memoryview, range],
    abc.Set: [set, frozenset],
    abc.Sized: [set, dict, frozenset, list, tuple, str, bytes, bytearray, memoryview, range],
}  # fmt: skip


@pytest.mark.parametrize("kind", BUILTINS)
def test_abc(kind: type) -> None:
    r"""Test that the builtins satisfy the protocols."""
    assert kind in BUILTINS, f"{kind} is not in BUILTINS"
    for cls in BUILTINS[kind]:
        assert issubclass(cls, kind), f"{cls} does not satisfy {kind}"


def show_abc() -> None:
    r"""Test which builtins satisfy which protocols."""
    supports = defaultdict(list)
    for cls in CLASSES:
        for proto in ABCS:
            if issubclass(cls, proto):
                supports[cls.__name__].append(proto.__name__)

    pprint.pprint(supports)

    print("-" * 80)

    # reverse the dictionary:
    supported_by = defaultdict(list)
    for key, value in supports.items():
        for v in value:
            supported_by[v].append(key)

    pprint.pprint(dict(supported_by))
