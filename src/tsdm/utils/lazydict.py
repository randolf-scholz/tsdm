r"""A Lazy Dictionary implementation.

The LazyDict is a dictionary that is initialized with functions as the values.
Once the value is accessed, the function is called and the result is stored.
"""
from __future__ import annotations

__all__ = [
    # Classes
    "LazyDict",
    "LazyValue",
]


import warnings
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from itertools import chain
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeAlias, Union, overload

from typing_extensions import Self

from tsdm.types.variables import any2_var as T2
from tsdm.types.variables import any_var as T
from tsdm.types.variables import key2_var as K2
from tsdm.types.variables import key_var as K
from tsdm.types.variables import return_var_co as R
from tsdm.utils.funcutils import (
    get_function_args,
    get_return_typehint,
    is_positional_arg,
)
from tsdm.utils.strings import repr_mapping

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem


class LazyValue(Generic[R]):
    r"""A placeholder for uninitialized values."""

    func: Callable[..., R]
    args: Iterable[Any]
    kwargs: Mapping[str, Any]
    type_hint: str

    def __init__(
        self,
        func: Callable[..., R],
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

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        r"""Execute the function and return the result."""
        return self.func(*chain(self.args, args), **(self.kwargs | kwargs))

    def __repr__(self) -> str:
        r"""Return a string representation of the function."""
        return f"{self.__class__.__name__}<{self.type_hint}>"


FuncSpec: TypeAlias = Union[
    LazyValue,
    Callable[[], R],
    Callable[[T], R],
    tuple[LazyValue],
    tuple[Callable[[], R]],  # no args
    tuple[Callable[[T], R]],  # key arg
    tuple[Callable[..., R], tuple],  # args
    tuple[Callable[..., R], dict],  # kwargs
    tuple[Callable[..., R], tuple, dict],  # args, kwargs
]
"""A type alias for the possible values of a LazyDict."""


class LazyDict(dict[K, T]):
    r"""A Lazy Dictionary implementation.

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

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        r"""Create a new instance of the class."""
        # inherit Mixin Methods from MutableMapping
        # This is crucial because dict.get does not call __getitem__
        # Reference: https://stackoverflow.com/a/2390997/9318372
        cls.get = MutableMapping.get  # type: ignore[assignment, method-assign]
        cls.clear = MutableMapping.clear  # type: ignore[method-assign]
        cls.pop = MutableMapping.pop  # type: ignore[method-assign]
        cls.popitem = MutableMapping.popitem  # type: ignore[method-assign]
        cls.setdefault = MutableMapping.setdefault  # type: ignore[method-assign]
        cls.update = MutableMapping.update  # type: ignore[method-assign]
        return super().__new__(cls, *args, **kwargs)  # type: ignore[type-var]

    @overload
    def __init__(self, /, **kwargs: FuncSpec | T) -> None:
        ...

    @overload
    def __init__(
        self,
        mapping: Mapping[K, FuncSpec | T],
        /,
        **kwargs: FuncSpec | T,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        iterable: Iterable[tuple[K, FuncSpec | T]],
        /,
        **kwargs: FuncSpec | T,
    ) -> None:
        ...

    def __init__(self, /, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the dictionary."""
        super().__init__()
        MutableMapping.update(self, *args, **kwargs)

    def __getitem__(self, key: K) -> T:
        r"""Get the value of the key."""
        value = super().__getitem__(key)
        if isinstance(value, LazyValue):
            new_value = value()
            super().__setitem__(key, new_value)
            return new_value
        return value

    def __setitem__(self, key: K, value: FuncSpec | T) -> None:
        r"""Set the value of the key."""
        super().__setitem__(key, self._make_lazy_function(key, value))  # type: ignore[assignment]

    def __repr__(self) -> str:
        r"""Return the representation of the dictionary."""
        return repr_mapping(self)

    def __or__(self, other: Mapping[K2, T2], /) -> LazyDict[K | K2, T | T2]:
        new = self.copy()
        new.update(other)  # type: ignore[arg-type]
        return new  # type: ignore[return-value]

    def __ror__(self, other: Mapping[K2, T2], /) -> LazyDict[K | K2, T | T2]:
        if isinstance(other, self.__class__):
            return other | self
        warnings.warn(
            "Using __ror__ with a non-LazyDict is not recommended, "
            "It causes all values to be evaluated.",
            category=RuntimeWarning,
            source=LazyDict,
            stacklevel=2,
        )
        new = other.copy() if isinstance(other, LazyDict) else LazyDict(other)
        new.update(self.asdict())
        return new

    def __ior__(self: Self, other: SupportsKeysAndGetItem[K, T], /) -> Self:  # type: ignore[override, misc]
        # TODO: fix typing error
        self.update(other)
        return self

    def asdict(self) -> dict[K, T]:
        r"""Return a dictionary with all values evaluated."""
        return {k: self[k] for k in self}

    @staticmethod
    def _make_lazy_function(
        key: K,
        value: FuncSpec | T,
    ) -> LazyValue:
        match value:
            case LazyValue():
                return value
            case Callable():  # type: ignore[misc]
                args = get_function_args(value, mandatory=True)  # type: ignore[unreachable]
                match nargs := len(args):
                    case 0:
                        return LazyValue(func=value)
                    case 1 if all(is_positional_arg(p) for p in args):
                        # set the key as input
                        return LazyValue(func=value, args=(key,))
                    case _:
                        raise TypeError(f"Function {value} requires {nargs} args.")
            case [Callable()]:  # type: ignore[misc]
                return LazyDict._make_lazy_function(key, value[0])  # type: ignore[index]
            case Callable(), tuple():  # type: ignore[misc]
                return LazyValue(func=value[0], args=value[1])  # type: ignore[index, misc]
            case Callable(), dict():  # type: ignore[misc]
                return LazyValue(func=value[0], kwargs=value[1])  # type: ignore[index, arg-type, misc]
            case Callable(), tuple(), dict():  # type: ignore[misc]
                return LazyValue(value[0], args=value[1], kwargs=value[2])  # type: ignore[index, misc]
            case _:
                warnings.warn(
                    f"Value {value} is not a callable, returning as is."
                    f"Provide a tuple (func, args, kwargs) to create Lazy Entry",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return LazyValue(lambda: value)

    def copy(self) -> Self:
        r"""Return a shallow copy of the dictionary."""
        return self.__class__(self.items())
