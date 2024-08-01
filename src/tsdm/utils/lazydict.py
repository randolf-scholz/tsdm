r"""A Lazy Dictionary implementation.

The LazyDict is a dictionary that is initialized with functions as the values.
Once the value is accessed, the function is called and the result is stored.
"""

__all__ = [
    # Type Alias
    "LazySpec",
    # Classes
    "LazyDict",
    "LazyValue",
]

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Never,
    Optional,
    Self,
    cast,
    overload,
)

from tsdm.constants import EMPTY_MAP
from tsdm.types.protocols import SupportsKeysAndGetItem
from tsdm.utils.decorators import pprint_repr
from tsdm.utils.funcutils import get_return_typehint

type MaybeLazy[V] = V | LazyValue[V]
type LazySpec[V] = (
    V                                       # direct value
    | LazyValue[V]                          # lazy value
    | tuple[Callable[..., V], tuple]        # args
    | tuple[Callable[..., V], dict]         # kwargs
    | tuple[Callable[..., V], tuple, dict]  # args, kwargs
)  # fmt: skip
r"""A type alias for the possible values of a `LazyDict`."""


@pprint_repr
@dataclass(slots=True, init=False)  # use slots since many instances might be created.
class LazyValue[R]:  # +R
    r"""A placeholder for uninitialized values."""

    func: Callable[..., R]
    args: Iterable[Any]
    kwargs: Mapping[str, Any]
    type_hint: str

    @classmethod
    def from_spec(cls, spec: LazySpec[R], /) -> "LazyValue[R]":
        r"""Create a LazyValue from a spec."""
        match spec:
            case LazyValue() as lazy_value:
                return lazy_value
            case Callable() as func, tuple(args):  # type: ignore[misc]
                return cls(func, args=args)  # type: ignore[has-type]
            case Callable() as func, dict(kwargs):  # type: ignore[misc]
                return cls(func, kwargs=kwargs)  # type: ignore[has-type]
            case Callable() as func, tuple(args), dict(kwargs):  # type: ignore[misc]
                return cls(func, args=args, kwargs=kwargs)  # type: ignore[has-type]
            case value:  # direct value
                return cls(lambda: value)  # type: ignore[arg-type, return-value]

    def __init__(
        self,
        func: Callable[..., R],
        /,
        *,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] = EMPTY_MAP,
        type_hint: Optional[str] = None,
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.type_hint = (
            get_return_typehint(self.func) if type_hint is None else type_hint
        )

    def __call__(self) -> R:
        r"""Execute the function and return the result."""
        return self.func(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        r"""Return a string representation of the function."""
        return f"{self.__class__.__name__}<{self.type_hint}>"


@pprint_repr
class LazyDict[K, V](dict[K, V]):
    r"""A Lazy Dictionary implementation.

    Note:
        - Getter methods `__getitem__`, `.pop`, `.get` trigger the lazy evaluation.
        - Iterator methods `.values` and `.items` do not!
        - Using `__setitem__` or `.setdefault` does not create `LazyValue` entries.
        - Use `.get_lazy` and `.set_lazy` to lookup/create `LazyValue` entries.

    Values are allowed to be one of the following:

    - LazyFunction
    - Callable that takes exactly 0 mandatory args
    - Callable that takes axactly 1 mandatory positional arg and no mandatory kwargs
      - In this case, the key will be used as the first argument
    - tuple of the form tuple[Callable] as above
    - tuple of the form tuple[Callable, tuple]
    - tuple of the form tuple[Callable, dict]
    - tuple of the form tuple[Callable, tuple, dict]
    """

    @classmethod
    def from_func(
        cls,
        iterable: Iterable[K],
        func: Callable[Concatenate[K, ...], V],
        /,
        *,
        args: tuple = (),
        kwargs: Mapping[str, Any] = EMPTY_MAP,
        type_hint: Optional[str] = None,
    ) -> "LazyDict[K, V]":
        r"""Create a new LazyDict from the keys."""
        type_hint = get_return_typehint(func) if type_hint is None else type_hint
        return cls({
            key: LazyValue(func, args=(key, *args), kwargs=kwargs, type_hint=type_hint)
            for key in iterable
        })

    if TYPE_CHECKING:
        # fmt: off
        @overload
        def __new__(cls, /) -> "LazyDict": ...
        @overload  # mapping only
        def __new__(cls, items: Mapping[K, LazySpec[V]], /) -> "LazyDict[K, V]": ...
        @overload  # iterable only
        def __new__(cls, items: Iterable[tuple[K, LazySpec[V]]], /) -> "LazyDict[K, V]": ...
        @overload  # kwargs only
        def __new__(cls, /, **kwargs: LazySpec[V]) -> "LazyDict[str, V]": ...
        @overload  # mapping and kwargs
        def __new__(cls, items: Mapping[str, LazySpec[V]], /, **kwargs: LazySpec[V]) -> "LazyDict[str, V]": ...
        @overload  # iterable and kwargs
        def __new__(cls, items: Iterable[tuple[str, LazySpec[V]]], /, **kwargs: LazySpec[V]) -> "LazyDict[str, V]": ...
        # fmt: on

    def __init__(
        self,
        items: SupportsKeysAndGetItem[K, LazySpec[V]]
        | Iterable[tuple[K, LazySpec[V]]] = EMPTY_MAP,
        /,
        **kwargs: LazySpec[V],
    ) -> None:
        r"""Initialize the dictionary."""
        super().__init__()
        match items:
            case SupportsKeysAndGetItem() as lookup:
                for key in lookup.keys():  # noqa: SIM118
                    self.set_lazy(key, lookup[key])
            case Iterable() as iterable:
                for key, value in iterable:
                    self.set_lazy(key, value)
            case _:
                raise TypeError(f"Invalid type for {items=}")

        for key, value in kwargs.items():
            self.set_lazy(key, value)  # pyright: ignore[reportArgumentType]

    def __getitem__(self, key: K, /) -> V:
        r"""Get the value of the key."""
        value = super().__getitem__(key)
        if isinstance(value, LazyValue):
            new_value = value()
            super().__setitem__(key, new_value)
            return new_value
        return value

    def __or__[K2, T](self, other: Mapping[K2, T], /) -> "LazyDict[K | K2, V | T]":
        new = cast(LazyDict[K | K2, V | T], self.copy())
        new.update(other)  # type: ignore[arg-type]
        return new

    def __ror__[K2, T](self, other: Mapping[K2, T], /) -> Never:
        raise NotImplementedError(
            "Using __ror__ with a non-LazyDict is not supported,"
            " since it would causes all values to be evaluated.",
        )

    def __ior__(self, other: "SupportsKeysAndGetItem[K, V]", /) -> Self:  # type: ignore[override, misc]
        # TODO: fix typing error
        self.update(other)
        return self

    def asdict(self) -> dict[K, V]:
        r"""Return a dictionary with all values evaluated."""
        return {k: self[k] for k in self}

    def copy(self) -> Self:
        r"""Return a shallow copy of the dictionary."""
        new = self.__class__()
        new.update(self)
        return new

    @overload  # type: ignore[override]
    def get(self, key: K, /) -> Optional[V]: ...
    @overload
    def get(self, key: K, default: V, /) -> V: ...
    @overload
    def get[T](self, key: K, default: T, /) -> V | T: ...
    def get(self, key, default=None, /):
        r"""Get the value of the key."""
        try:
            return self[key]
        except KeyError:
            return default

    @overload
    def get_lazy(self, key: K, /) -> Optional[MaybeLazy[V]]: ...
    @overload
    def get_lazy(self, key: K, default: V, /) -> MaybeLazy[V]: ...
    @overload
    def get_lazy[T](self, key: K, default: T, /) -> MaybeLazy[V] | T: ...
    def get_lazy(self, key, default=None, /):
        r"""Get the value for the key lazily."""
        return super().get(key, default)

    def set_lazy(self, key: K, value: LazySpec[V], /) -> None:
        r"""Set the value directly."""
        lazy_value = LazyValue.from_spec(value)
        super().__setitem__(key, lazy_value)  # type: ignore[assignment]

    __NOTGIVEN = object()

    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop[T](self, key: K, default: T, /) -> V | T: ...
    def pop(self, key, default=__NOTGIVEN, /):
        r"""Pop the value of the key."""
        value = (
            super().pop(key)
            if default is self.__NOTGIVEN
            else super().pop(key, default)
        )
        if isinstance(value, LazyValue):
            return value()
        return value

    def popitem(self) -> tuple[K, V]:
        r"""Pop the last item."""
        key, value = super().popitem()
        if isinstance(value, LazyValue):
            return key, cast(V, value())
        return key, value
