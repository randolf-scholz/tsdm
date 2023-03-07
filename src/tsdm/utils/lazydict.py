r"""A Lazy Dictionary implementation.

The LazyDict is a dictionary that is initialized with functions as the values.
Once the value is accessed, the function is called and the result is stored.
"""
from __future__ import annotations

__all__ = [
    # Classes
    "LazyDict",
    "LazyFunction",
]


import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, Union, cast, overload

from typing_extensions import Self

from tsdm.types.variables import Any2Var as T2
from tsdm.types.variables import AnyVar as T
from tsdm.types.variables import Key2Var as K2
from tsdm.types.variables import KeyVar as K
from tsdm.types.variables import ReturnVar_co as R
from tsdm.utils._utils import get_function_args, is_positional_arg
from tsdm.utils.strings import repr_mapping

__logger__ = logging.getLogger(__name__)

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem


class LazyFunction(Generic[R]):
    r"""A placeholder for uninitialized values."""

    # TODO: Check for Type Hints and add it to __repr__
    # FIXME: use typing.NamedTuple (3.11)
    func: Callable[..., R]
    args: Iterable[Any]
    kwargs: Mapping[str, Any]

    def __init__(
        self,
        func: Callable[..., R],
        *,
        args: Iterable[Any] = NotImplemented,
        kwargs: Mapping[str, Any] = NotImplemented,
    ) -> None:
        self.func = func
        self.args = args if args is not NotImplemented else ()
        self.kwargs = kwargs if kwargs is not NotImplemented else {}

    def __call__(self) -> R:
        r"""Execute the function and return the result."""
        return self.func(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        r"""Return a string representation of the function."""
        if isinstance(self.func, FunctionType | MethodType):
            ann = self.func.__annotations__.get("return", object)  # type: ignore[unreachable]
        else:
            ann = self.func.__call__.__annotations__.get("return", object)  # type: ignore[operator]

        # TODO: pretty print annotations, especially for typing types
        if isinstance(ann, type):
            val = ann.__name__
        else:
            val = str(ann)

        return f"LazyFunction<{val}>"


FuncSpec: TypeAlias = Union[
    LazyFunction,
    Callable[[], R],
    Callable[[T], R],
    tuple[LazyFunction],
    tuple[Callable[[], R]],  # no args
    tuple[Callable[[T], R]],  # key arg
    tuple[Callable[..., R], tuple],  # args
    tuple[Callable[..., R], dict],  # kwargs
    tuple[Callable[..., R], tuple, dict],  # args, kwargs
]


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

    def __new__(cls, *args, **kwargs):
        r"""Create a new instance of the class."""
        # inherit Mixin Methods from MutableMapping
        # This is crucial because dict.get does not call __getitem__
        # Reference: https://stackoverflow.com/a/2390997/9318372
        cls.get = MutableMapping.get  # type: ignore[assignment]
        cls.clear = MutableMapping.clear  # type: ignore[assignment]
        cls.pop = MutableMapping.pop  # type: ignore[assignment]
        cls.popitem = MutableMapping.popitem  # type: ignore[assignment]
        cls.setdefault = MutableMapping.setdefault  # type: ignore[assignment]
        cls.update = MutableMapping.update  # type: ignore[assignment]
        return super().__new__(cls, *args, **kwargs)

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
        if isinstance(value, LazyFunction):
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
        return cast(LazyDict[K | K2, T | T2], new)

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
    ) -> LazyFunction:
        match value:
            case LazyFunction():
                return value
            case Callable():  # type: ignore[misc]
                args = get_function_args(value, mandatory=True)  # type: ignore[unreachable]
                match nargs := len(args):
                    case 0:
                        return LazyFunction(func=value)
                    case 1 if all(
                        is_positional_arg(p) for p in args
                    ):  # set the key as input
                        return LazyFunction(func=value, args=(key,))
                    case _:
                        raise TypeError(f"Function {value} requires {nargs} args.")
            case [Callable()]:  # type: ignore[misc]
                return LazyDict._make_lazy_function(key, value[0])  # type: ignore[index]
            case Callable(), tuple():  # type: ignore[misc]
                return LazyFunction(func=value[0], args=value[1])  # type: ignore[index, misc]
            case Callable(), dict():  # type: ignore[misc]
                return LazyFunction(func=value[0], kwargs=value[1])  # type: ignore[index, arg-type, misc]
            case Callable(), tuple(), dict():  # type: ignore[misc]
                return LazyFunction(value[0], args=value[1], kwargs=value[2])  # type: ignore[index, misc]
            case _:
                return LazyFunction(lambda: value)

    def copy(self) -> Self:
        r"""Return a shallow copy of the dictionary."""
        return self.__class__(self.items())
