"""Test which builtins satisfy which protocols."""

import pprint
from collections import abc, defaultdict

ABCS: list[type] = [
    abc.Container,
    abc.Hashable,
    abc.Iterable,
    abc.Iterator,
    abc.Generator,
    abc.Reversible,
    abc.Sized,
    abc.Callable,  # type: ignore[list-item]
    abc.Collection,
    abc.Sequence,
    abc.MutableSequence,
    abc.ByteString,  # type: ignore[list-item]
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


def test_abc() -> None:
    """Test which builtins satisfy which protocols."""
    supports = defaultdict(list)
    for cls in CLASSES:
        for proto in ABCS:
            if issubclass(cls, proto):
                supports[str(cls)].append(proto.__name__)

    pprint.pprint(supports)

    print("-" * 80)

    # reverse the dictionary:
    supported_by = defaultdict(list)
    for key, value in supports.items():
        for v in value:
            supported_by[v].append(key)

    pprint.pprint(supported_by)
