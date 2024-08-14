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
    LazyValue[V]                                        # LazyValue
    | tuple[Callable[..., V], tuple, dict[str, Any]]      # func, args, kwargs
    | Callable[[], V]                                     # func
    # | V                                                 # value (cannot be tuple)
)  # fmt: skip
r"""A type alias for the possible values of a `LazyDict`."""


@pprint_repr
@dataclass(slots=True, init=False)  # use slots since many instances might be created.
class LazyValue[V]:  # +V
    r"""A placeholder for uninitialized values."""

    func: Callable[..., V]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    type_hint: str

    @staticmethod
    def unwrap[T](value: MaybeLazy[T], /) -> T:
        r"""Unwrap the value if it is a LazyValue."""
        if isinstance(value, LazyValue):
            return value.compute()
        return value

    @classmethod
    def from_spec(cls, spec: LazySpec[V], /) -> "LazyValue[V]":
        r"""Create a LazyValue from a spec."""
        match spec:
            case LazyValue() as lazy_value:
                return lazy_value
            case tuple([Callable() as func, tuple(args), dict(kwargs)]):  # type: ignore[misc]
                return cls(func, args=args, kwargs=kwargs)  # type: ignore[has-type]
            case tuple(tup):
                raise TypeError(
                    "Expected tuple of function, args and kwargs."
                    f" Got tuple[{', '.join(type(arg).__name__ for arg in tup)}].",
                )
            case Callable() as func:  # type: ignore[misc]
                return cls(func)  # type: ignore[unreachable]
            case value:  # Fallback: wrap value.
                raise TypeError(
                    f"Expected LazyValue or Callable. Got {type(value).__name__}."
                )
                # return cls(lambda: value, type_hint=type(value).__name__)  # type: ignore[arg-type, return-value]

    def __init__(
        self,
        func: Callable[..., V],
        /,
        *,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] = EMPTY_MAP,
        type_hint: Optional[str] = None,
    ) -> None:
        self.func = func
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.type_hint = (
            get_return_typehint(self.func) if type_hint is None else type_hint
        )

    def compute(self) -> V:
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
        r"""Create a new LazyDict by passing the keys to a function.

        Args:
            iterable: The keys to use.
            func: The function to use. The first argument is the key.
            args: Additonal fixed positional arguments to pass to the function.
            kwargs: Additonal fixed keyword arguments to pass to the function.
            type_hint: The type hint of the values. Default: infer from `func`.
        """
        type_hint = get_return_typehint(func) if type_hint is None else type_hint

        return cls({
            key: LazyValue(func, args=(key, *args), kwargs=kwargs, type_hint=type_hint)
            for key in iterable
        })

    if TYPE_CHECKING:

        @overload
        def __new__(cls, /) -> "LazyDict": ...
        @overload  # mapping only
        def __new__(cls, items: Mapping[K, LazySpec[V]], /) -> "LazyDict[K, V]": ...
        @overload  # mapping and kwargs
        def __new__(
            cls, items: Mapping[str, LazySpec[V]] = ..., /, **kwargs: LazySpec[V]
        ) -> "LazyDict[str, V]": ...

    def __init__(
        self, mapping: Mapping[K, LazySpec[V]] = EMPTY_MAP, /, **kwargs: LazySpec[V]
    ) -> None:
        r"""Initialize the dictionary."""
        super().__init__()
        for key, value in mapping.items():
            self.set_lazy(key, value)
        for key2, value2 in kwargs.items():
            self.set_lazy(key2, value2)  # type: ignore[arg-type]

    def __getitem__(self, key: K, /) -> V:
        r"""Get the value of the key."""
        value = super().__getitem__(key)
        unwrapped_value = LazyValue.unwrap(value)
        if unwrapped_value is not value:
            super().__setitem__(key, unwrapped_value)
        return unwrapped_value

    # NOTE: Overloaded operator methods can't have wider argument types in overrides [mypy 1.11]
    @overload
    def __or__(self, other: dict[K, V], /) -> Self: ...
    @overload
    def __or__[K2, V2](self, other: dict[K2, V2], /) -> "LazyDict[K | K2, V | V2]": ...
    def __or__[K2, V2](self, other: Mapping[K2, V2], /) -> "LazyDict[K | K2, V | V2]":
        new = cast(LazyDict[K | K2, V | V2], self.copy())
        new.update(other)  # type: ignore[arg-type]
        return new

    def __ror__[K2, T](self, other: Mapping[K2, T], /) -> Never:
        raise NotImplementedError(
            "Using __ror__ with a non-LazyDict is not supported,"
            " since it would causes all values to be evaluated.",
        )

    def __ior__(self, other: SupportsKeysAndGetItem[K, V], /) -> Self:  # type: ignore[override, misc]
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

    # NOTE: need to overwrite since dict.get does not call __getitem__.
    #   Also, dict.get has different overloads than Mapping.get.
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
        r"""Set the value wrapped as LazyValue."""
        lazy_value = LazyValue.from_spec(value)
        super().__setitem__(key, lazy_value)  # type: ignore[assignment]

    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop[T](self, key: K, default: T, /) -> V | T: ...
    def pop[T](self, key: K, /, *args: *tuple[V | T, ...]) -> V | T:
        r"""Pop the value of the key."""
        value = super().pop(key, *args)
        return LazyValue.unwrap(value)

    def popitem(self) -> tuple[K, V]:
        r"""Pop the last item."""
        key, value = super().popitem()
        return key, LazyValue.unwrap(value)
