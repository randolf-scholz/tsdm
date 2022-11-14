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
from typing import Any, NamedTuple, TypeAlias, Union, overload

from tsdm.utils._utils import get_function_args, is_positional
from tsdm.utils.strings import repr_mapping, repr_short
from tsdm.utils.types import KeyVar, ObjectVar

__logger__ = logging.getLogger(__name__)


class LazyFunction(NamedTuple):
    r"""A placeholder for uninitialized values."""

    func: Callable[..., Any]
    args: Iterable[Any] = ()
    kwargs: Mapping[str, Any] = {}

    def __call__(self) -> Any:
        r"""Execute the function and return the result."""
        return self.func(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        r"""Return a string representation of the function."""
        return f"<LazyFunction: {self.func.__name__}>"


FuncSpec: TypeAlias = Union[
    LazyFunction,
    Callable[[], ObjectVar],
    Callable[[KeyVar], ObjectVar],
    tuple[LazyFunction],
    tuple[Callable[[], ObjectVar]],  # no args
    tuple[Callable[[KeyVar], ObjectVar]],  # key arg
    tuple[Callable[..., ObjectVar], tuple],  # args
    tuple[Callable[..., ObjectVar], dict],  # kwargs
    tuple[Callable[..., ObjectVar], tuple, dict],  # args, kwargs
]


class LazyDict(dict[KeyVar, ObjectVar]):
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
    def __init__(self, /, **kwargs: FuncSpec | ObjectVar) -> None:
        ...

    @overload
    def __init__(
        self,
        mapping: Mapping[KeyVar, FuncSpec | ObjectVar],
        /,
        **kwargs: FuncSpec | ObjectVar,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        iterable: Iterable[tuple[KeyVar, FuncSpec | ObjectVar]],
        /,
        **kwargs: FuncSpec | ObjectVar,
    ) -> None:
        ...

    def __init__(self, /, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the dictionary."""
        super().__init__()
        MutableMapping.update(self, *args, **kwargs)

    def __getitem__(self, key: KeyVar) -> ObjectVar:
        r"""Get the value of the key."""
        value = super().__getitem__(key)
        if isinstance(value, LazyFunction):
            new_value = value()
            super().__setitem__(key, new_value)
            return new_value
        return value

    def __setitem__(self, key: KeyVar, value: FuncSpec | ObjectVar) -> None:
        r"""Set the value of the key."""
        super().__setitem__(key, self._make_lazy_function(key, value))  # type: ignore[assignment]

    def __repr__(self) -> str:
        r"""Return the representation of the dictionary."""
        return repr_mapping(self, repr_fun=repr_short)

    def __or__(self, other, /):
        # FIXME: https://github.com/python/cpython/issues/99327
        # TODO: Self python 3.11
        new = self.copy()
        new.update(other)
        return new

    def __ror__(self, other, /):
        # FIXME: https://github.com/python/cpython/issues/99327
        # TODO: Self python 3.11
        if isinstance(other, self.__class__):
            return other | self
        warnings.warn(
            "Using __ror__ with a non-LazyDict is not recommended, "
            "It causes all values to be evaluated.",
            category=RuntimeWarning,
            source=LazyDict,
        )
        new = other.copy()
        new.update(self.asdict())
        return new

    def __ior__(self, other, /):
        # FIXME: https://github.com/python/cpython/issues/99327
        # TODO: Self python 3.11
        self.update(other)
        return self

    def asdict(self) -> dict[KeyVar, ObjectVar]:
        r"""Return a dictionary with all values evaluated."""
        return {k: self[k] for k in self}

    @staticmethod
    def _make_lazy_function(
        key: KeyVar,
        value: FuncSpec | ObjectVar,
    ) -> LazyFunction:
        match value:
            case LazyFunction():
                return value  # type: ignore[return-value]
            case Callable():  # type: ignore[misc]
                args = get_function_args(value, mandatory=True)  # type: ignore[unreachable]
                match nargs := len(args):
                    case 0:
                        return LazyFunction(func=value)
                    case 1 if all(
                        is_positional(p) for p in args
                    ):  # set the key as input
                        return LazyFunction(func=value, args=(key,))
                    case _:
                        raise TypeError(f"Function {value} requires {nargs} arguments.")
            case [Callable()]:  # type: ignore[misc]
                return LazyDict._make_lazy_function(key, value[0])  # type: ignore[index]
            case Callable(), tuple():  # type: ignore[misc]
                return LazyFunction(func=value[0], args=value[1])  # type: ignore[index, arg-type, misc]
            case Callable(), dict():  # type: ignore[misc]
                return LazyFunction(func=value[0], kwargs=value[1])  # type: ignore[index, arg-type, misc]
            case Callable(), tuple(), dict():  # type: ignore[misc]
                return LazyFunction(value[0], args=value[1], kwargs=value[2])  # type: ignore[index,  arg-type, misc]
            case _:
                return LazyFunction(lambda: value)

    def copy(self) -> LazyDict[KeyVar, ObjectVar]:
        r"""Return a shallow copy of the dictionary."""
        return self.__class__(self.items())
