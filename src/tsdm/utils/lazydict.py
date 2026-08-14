r"""A Lazy Dictionary implementation.

The LazyDict is a dictionary initialized with functions as the values.
Once the value is accessed, the function is called and the result is stored.
"""

__all__ = [
    # Type Alias
    "Lazy",
    # Classes
    "LazyDict",
    "LazyValue",
    # Functions
    "lazy_dict",
]

from collections.abc import Callable, ItemsView, Iterable, Mapping, ValuesView
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Never,
    Optional,
    Self,
    overload,
)

from tsdm.pprint import pprint_repr

from .funcutils import get_return_typehint

type Lazy[V] = Callable[[], V]


@pprint_repr
@dataclass(slots=True, init=False)  # use slots since many instances might be created.
class LazyValue[V]:  # +V
    r"""A placeholder for uninitialized values."""

    func: Callable[..., V]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    type_hint: str

    # private
    __marker: ClassVar[object] = object()  # marker for uninitialized value
    _value: Any

    @property
    def value(self) -> V:
        if self._value is self.__marker:
            try:
                value = self.func(*self.args, **self.kwargs)
            except Exception as exc:
                exc.add_note("Failed to evaluate LazyValue.")
                raise
            self._value = value
        return self._value

    def __call__(self) -> V:  # for compatibility with Callable[[], V]
        return self.value

    def __init__(
        self,
        func: Callable[..., V],
        /,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        type_hint: Optional[str] = None,
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.type_hint = (
            get_return_typehint(self.func) if type_hint is None else type_hint
        )
        self._value = self.__marker

        # validation
        if isinstance(func, LazyValue) and (self.args or self.kwargs):
            raise ValueError("Got unexpected args or kwargs for LazyValue.")

    @staticmethod
    def unwrap[T](arg: T | LazyValue[T], /) -> T:
        r"""Unwrap the value if it is a LazyValue."""
        if isinstance(arg, LazyValue):
            # recursion to unwrap nested LazyValues
            # TODO: check for fix-points?
            return LazyValue.unwrap(arg.value)
        return arg

    def __repr__(self) -> str:
        r"""Return a string representation of the function."""
        return f"{self.__class__.__name__}<{self.type_hint}>"


@pprint_repr
class LazyDict[K = Any, V = Any](dict[K, V]):
    r"""A Lazy Dictionary implementation.

    Note:
        - Getter methods `__getitem__`, `.pop`, `.get` trigger the lazy evaluation.
        - View-operations `.values()` and `.items()` do not!
        - Using `__setitem__` or `.setdefault` does not create `LazyValue` entries.
        - Use `.get_lazy` and `.set_lazy` to lookup/create `LazyValue` entries.

    Values are allowed to be one of the following:

    - LazyFunction
    - Callable that takes exactly 0 mandatory args
    - Callable that takes exactly 1 mandatory positional arg and no mandatory kwargs
      - In this case, the key will be used as the first argument
    - tuple of the form tuple[Callable] as above
    - tuple of the form tuple[Callable, tuple]
    - tuple of the form tuple[Callable, dict]
    - tuple of the form tuple[Callable, tuple, dict]
    """

    @overload
    @staticmethod
    def new[T = Any, X = Any](  # pyright: ignore[reportOverlappingOverload]
        items: Mapping[T, Lazy[X]] | Iterable[tuple[T, Lazy[X]]] = ...,  # pyright: ignore[reportInvalidTypeVarUse]
        /,
    ) -> LazyDict[T, X]: ...
    @overload  # mapping and kwargs
    @staticmethod
    def new[T = Never, X = Any](
        items: Mapping[T, Lazy[X]] | Iterable[tuple[T, Lazy[X]]] = ...,  # pyright: ignore[reportInvalidTypeVarUse]
        /,
        **kwargs: Lazy[X],
    ) -> LazyDict[T | str, X]: ...
    @staticmethod
    def new[T = Never, X = Any](
        args: Mapping[T, Lazy[X]] | Iterable[tuple[T, Lazy[X]]] = (),
        /,
        **kwargs: Lazy[X],
    ) -> LazyDict[T, X] | LazyDict[T | str, X]:
        r"""Create a new LazyDict."""
        self = LazyDict[Any, X]()

        for key, value in dict(args).items():
            self.set_lazy(key, value)

        for str_key, value in kwargs.items():
            self.set_lazy(str_key, value)

        return self

    @staticmethod
    def from_func(
        iterable: Iterable[K],
        func: Callable[Concatenate[K, ...], V],
        /,
        *,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        type_hint: Optional[str] = None,
    ) -> LazyDict[K, V]:
        r"""Create a new LazyDict by passing the keys to a function.

        Args:
            iterable: The keys to use.
            func: The function to use. The first argument is the key.
            args: Additonal fixed positional arguments to pass to the function.
            kwargs: Additonal fixed keyword arguments to pass to the function.
            type_hint: The type hint of the values. Default: infer from `func`.
        """
        type_hint = get_return_typehint(func) if type_hint is None else type_hint

        return LazyDict.new(
            {
                key: LazyValue(
                    func, args=(key, *args), kwargs=kwargs, type_hint=type_hint
                )
                for key in iterable
            }
        )

    if TYPE_CHECKING:
        # fmt: off
        def values(self) -> ValuesView[V | LazyValue[V]]: ...  # type: ignore
        def items(self) -> ItemsView[K, V | LazyValue[V]]: ...  # type: ignore
        # fmt: on

    def __getitem__(self, key: K, /) -> V:
        r"""Get the value of the key."""
        value = super().__getitem__(key)
        unwrapped_value = LazyValue.unwrap(value)
        if unwrapped_value is not value:
            super().__setitem__(key, unwrapped_value)
        return unwrapped_value

    # @overload
    # def get_lazy(self, key: K, /) -> Optional[V | LazyValue[V]]: ...
    # @overload
    # def get_lazy[T](self, key: K, default: T, /) -> T | V | LazyValue[V]: ...
    def get_lazy[T](
        self, key: K, default: Optional[T] = None, /
    ) -> V | LazyValue[V] | T | None:
        r"""Get the value for the key lazily."""
        return super().get(key, default)

    def set_lazy(
        self,
        key: K,
        value: Lazy[V],
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        r"""Set the value wrapped as LazyValue."""
        lazy_value = LazyValue(value, args=args, kwargs=kwargs)
        super().__setitem__(key, lazy_value)  # type: ignore

    def asdict(self) -> dict[K, V]:
        r"""Return a dictionary with all values evaluated."""
        return {k: self[k] for k in self}

    # region dict-methods --------------------------------------------------------------

    def __or__[K2, V2](self, other: Mapping[K2, V2], /) -> LazyDict[K | K2, V | V2]:
        new_dict = super().__or__(dict(other))
        return LazyDict(new_dict)

    def __ror__[K2, V2](self, other: Mapping[K2, V2], /) -> LazyDict[K | K2, V | V2]:
        new_dict = super().__ror__(dict(other))
        return LazyDict(new_dict)

    def copy(self) -> Self:
        r"""Return a shallow copy of the dictionary."""
        return self.__class__(super().copy())

    # NOTE: need to overwrite since dict.get does not call __getitem__.
    #   Also, dict.get has different overloads than Mapping.get.
    @overload
    def get(self, key: K, /, default: None = ...) -> V | None: ...
    @overload
    def get[T](self, key: K, /, default: T) -> V | T: ...
    def get[T](self, key: K, /, default: Optional[T | V] = None) -> V | T | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""Get the value of the key."""
        try:
            return self[key]
        except KeyError:
            return default

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

    # endregion dict-methods -----------------------------------------------------------


@overload  # mapping only
def lazy_dict[K = Any, V = Any](  # pyright: ignore[reportOverlappingOverload]
    items: Mapping[K, Lazy[V]] | Iterable[tuple[K, Lazy[V]]] = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    /,
) -> LazyDict[K, V]: ...
@overload  # mapping and kwargs
def lazy_dict[K = Never, V = Any](
    items: Mapping[K, Lazy[V]] | Iterable[tuple[K, Lazy[V]]] = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    /,
    **kwargs: Lazy[V],
) -> LazyDict[K | str, V]: ...
def lazy_dict[K = Never, V = Any](
    arg: Mapping[K, Lazy[V]] | Iterable[tuple[K, Lazy[V]]] = (),
    /,
    **kwargs: Lazy[V],
) -> LazyDict[K, V] | LazyDict[K | str, V]:
    r"""Create a new LazyDict from an iterable of keys and a Lazy."""
    return LazyDict.new(arg, **kwargs)
