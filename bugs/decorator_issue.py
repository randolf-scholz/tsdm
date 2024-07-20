from collections.abc import Iterator, Mapping
from functools import partial, partialmethod, wraps
from typing import Protocol, overload


class BareClassDecorator[Cls: type](Protocol):
    def __call__(self, cls: Cls, /) -> Cls: ...


class ClassDecorator[Cls: type, **P](Protocol):
    def __call__(self, cls: Cls, /, *args: P.args, **kwargs: P.kwargs) -> Cls: ...


class ParametrizedClassDecorator[Cls: type, **P](Protocol):
    @overload  # @decorator(*args, **kwargs)
    def __call__(
        self, /, *args: P.args, **kwargs: P.kwargs
    ) -> BareClassDecorator[Cls]: ...
    @overload  # @decorator
    def __call__(self, cls: Cls, /, *args: P.args, **kwargs: P.kwargs) -> Cls: ...


def decorator[T, **P](
    deco: ClassDecorator[type[T], P], /
) -> ParametrizedClassDecorator[type[T], P]:
    __sentinel = object()

    @wraps(deco)
    def __parametrized_decorator(obj=__sentinel, /, *args, **kwargs):
        if obj is __sentinel:
            return partial(deco, *args, **kwargs)
        return deco(obj, *args, **kwargs)

    return __parametrized_decorator


# the actual pprint function
def repr_mapping(
    obj: Mapping, /, *, linebreaks: bool = False, maxitems: int = 6
) -> str: ...


# the parametrized decorator
@decorator
def pprint_mapping[Map: type[Mapping]](cls: Map, /, **kwds) -> Map:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Mapping):
        raise TypeError(f"Expected Sequence type, got {cls}.")
    cls.__repr__ = partialmethod(repr_mapping, **kwds)  # type: ignore[assignment]
    return cls


@pprint_mapping
class TestMapping(Mapping[str, int]):
    r"""Test Mapping."""

    def __getitem__(self, key: str, /) -> int:
        return int(key)

    def __iter__(self) -> Iterator[str]:
        return iter(map(str, range(5)))

    def __len__(self) -> int:
        return 10
