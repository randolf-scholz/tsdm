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

import warnings
from collections.abc import Callable, Iterable, Mapping

from typing_extensions import (
    Any,
    Generic,
    Optional,
    Self,
    TypeAlias,
    cast,
    overload,
)

from tsdm.constants import EMPTY_MAP
from tsdm.types.protocols import SupportsKeysAndGetItem
from tsdm.types.variables import K2, K, R, T, V
from tsdm.utils.funcutils import (
    get_function_args,
    get_return_typehint,
    is_positional_arg,
)
from tsdm.utils.pprint import pprint_repr


class LazyValue(Generic[R]):
    r"""A placeholder for uninitialized values."""

    __slots__ = ("args", "func", "kwargs", "type_hint")
    # we use slots since lots of instances of lazy-value might be created.

    func: Callable[..., R]
    args: Iterable[Any]
    kwargs: Mapping[str, Any]
    type_hint: str

    def __init__(
        self,
        func: Callable[..., R],
        /,
        *,
        args: Iterable[Any] = NotImplemented,
        kwargs: Mapping[str, Any] = NotImplemented,
        type_hint: Optional[str] = None,
    ) -> None:
        self.func = func
        self.args = args if args is not NotImplemented else ()
        self.kwargs = kwargs if kwargs is not NotImplemented else {}
        self.type_hint = (
            get_return_typehint(self.func) if type_hint is None else type_hint
        )

    def __call__(self) -> R:
        r"""Execute the function and return the result."""
        return self.func(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        r"""Return a string representation of the function."""
        return f"{self.__class__.__name__}<{self.type_hint}>"


LazySpec: TypeAlias = (
    LazyValue[T]  # lazy value
    | Callable[[], T]  # no args
    | Callable[[Any], T]  # single arg
    | tuple[Callable[..., T], tuple]  # args
    | tuple[Callable[..., T], dict]  # kwargs
    | tuple[Callable[..., T], tuple, dict]  # args, kwargs
    | T  # direct value
)
r"""A type alias for the possible values of a `LazyDict`."""


@pprint_repr
class LazyDict(dict[K, V]):
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

    __slots__ = ()

    # def __new__(cls, *args: Any, **kwargs: Any) -> Self:
    #     r"""Create a new instance of the class."""
    #     # inherit Mixin Methods from MutableMapping
    #     # This is crucial because dict.get does not call __getitem__
    #     # Reference: https://stackoverflow.com/a/2390997
    #     # cls.get = MutableMapping.get  # type: ignore[assignment, method-assign]
    #     # cls.clear = MutableMapping.clear  # type: ignore[method-assign]
    #     # cls.pop = MutableMapping.pop  # type: ignore[method-assign]
    #     # cls.popitem = MutableMapping.popitem  # type: ignore[method-assign]
    #     # cls.setdefault = MutableMapping.setdefault  # type: ignore[method-assign]
    #     cls.update = MutableMapping.update  # type: ignore[method-assign]
    #     return super().__new__(cls, *args, **kwargs)  # type: ignore[type-var]

    # inherit update from MutableMapping
    # update = MutableMapping.update

    @overload
    def __init__(self, /, **kwargs: LazySpec[V]) -> None: ...
    @overload
    def __init__(
        self, mapping: SupportsKeysAndGetItem[K, LazySpec[V]], /, **kwargs: LazySpec[V]
    ) -> None: ...
    @overload
    def __init__(
        self, iterable: Iterable[tuple[K, LazySpec[V]]], /, **kwargs: LazySpec[V]
    ) -> None: ...
    def __init__(self, other=EMPTY_MAP, /, **kwargs):  # pyright: ignore
        r"""Initialize the dictionary."""
        super().__init__()
        match other:
            case SupportsKeysAndGetItem() as lookup:
                for key in lookup.keys():  # noqa: SIM118
                    self.set_lazy(key, lookup[key])
            case Iterable() as iterable:
                for key, value in iterable:
                    self.set_lazy(key, value)
            case _:
                raise TypeError(f"Invalid type for {other=}")

        for key, value in kwargs.items():
            self.set_lazy(key, value)  # pyright: ignore

    def __getitem__(self, key: K, /) -> V:
        r"""Get the value of the key."""
        value = super().__getitem__(key)
        if isinstance(value, LazyValue):
            new_value = value()
            super().__setitem__(key, new_value)
            return new_value
        return value

    @overload
    def get_lazy(self, key: K, /) -> V | LazyValue[V] | None: ...
    @overload
    def get_lazy(self, key: K, /, *, default: V) -> V | LazyValue[V]: ...
    @overload
    def get_lazy(self, key: K, /, *, default: T) -> V | LazyValue[V] | T: ...
    def get_lazy(self, key, /, *, default=None):
        r"""Get the lazy value of the key."""
        return super().get(key, default)

    def set_lazy(self, key: K, value: LazySpec[V], /) -> None:
        r"""Set the value directly."""
        lazy_value = self._make_lazy_function(key, value)
        super().__setitem__(key, lazy_value)  # type: ignore[assignment]

    def __or__(self, other: Mapping[K2, T], /) -> "LazyDict[K | K2, V | T]":
        new = self.copy()
        new.update(other)  # type: ignore[arg-type]
        return new  # type: ignore[return-value]

    def __ror__(self, other: Mapping[K2, T], /) -> "LazyDict[K | K2, V | T]":
        if isinstance(other, self.__class__):
            return other | self  # pyright: ignore

        warnings.warn(
            "Using __ror__ with a non-LazyDict is not recommended, "
            "It causes all values to be evaluated.",
            category=RuntimeWarning,
            source=LazyDict,
            stacklevel=2,
        )

        new = cast("LazyDict[K | K2, V | T]", LazyDict(other))
        new.update(self)  # type: ignore[arg-type]
        return new

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
        new.update(self)  # type: ignore[arg-type]
        return new

    @overload  # type: ignore[override]
    def get(self, key: K, /) -> V | None: ...
    @overload
    def get(self, key: K, default: V, /) -> V: ...
    @overload
    def get(self, key: K, default: T, /) -> V | T: ...
    def get(self, key, default=None, /):
        r"""Get the value of the key."""
        try:
            return self[key]
        except KeyError:
            return default

    __NOTGIVEN = object()

    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, default: V, /) -> V: ...
    @overload
    def pop(self, key: K, default: T, /) -> V | T: ...
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

    @classmethod  # type: ignore[override]
    @overload
    def fromkeys(cls, iterable: Iterable[K], value: None = ..., /) -> Self: ...
    @classmethod
    @overload
    def fromkeys(cls, iterable: Iterable[K], value: LazySpec[V], /) -> Self: ...
    @classmethod
    def fromkeys(cls, iterable, value=None):
        r"""Create a new LazyDict from the keys."""
        return cls({k: value for k in iterable})

    @staticmethod
    def _make_lazy_function(key: K, value: LazySpec[V], /) -> LazyValue[V]:
        match value:
            case LazyValue():
                return value
            case Callable() as func:  # type: ignore[misc]
                args = get_function_args(func, mandatory=True)  # type: ignore[unreachable]
                match nargs := len(args):
                    case 0:
                        return LazyValue(func)
                    case 1 if all(is_positional_arg(p) for p in args):
                        return LazyValue(func, args=(key,))  # set the key as input
                    case _:
                        raise TypeError(f"Function {func} requires {nargs} args.")
            case Callable() as func, tuple() as args:  # type: ignore[misc]
                return LazyValue(func, args=args)  # type: ignore[has-type]
            case Callable() as func, dict() as kwargs:  # type: ignore[misc]
                return LazyValue(func, kwargs=kwargs)  # type: ignore[has-type]
            case Callable() as func, tuple() as args, dict() as kwargs:  # type: ignore[misc]
                return LazyValue(func, args=args, kwargs=kwargs)  # type: ignore[has-type]
            case _:
                warnings.warn(
                    f"Value {value} for key {key!r} is not a callable."
                    " Provide a tuple (func, args, kwargs) to create Lazy Entry."
                    " Wrapping the value in LazyValue instead."
                    " To set values directly, use the `set` method.",
                    category=RuntimeWarning,
                    source=LazyDict,
                    stacklevel=3,
                )
                value = cast(V, value)
                return LazyValue(lambda: value)
