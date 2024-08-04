r"""Submodule containing general purpose decorators."""

__all__ = [
    # Classes
    "BareDecorator",
    "BareFunctionDecorator",
    "BareClassDecorator",
    "ClassDecorator",
    "ClassDecoratorFactory",
    "Decorator",
    "DecoratorError",
    "DecoratorFactory",
    "FunctionDecorator",
    "FunctionDecoratorFactory",
    "ParametrizedClassDecorator",
    "ParametrizedDecorator",
    "ParametrizedFunctionDecorator",
    # Functions
    "attribute",
    "decorator",
    "recurse_on_container",
]

import logging
from collections.abc import Callable as Fn
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Optional, Protocol, Self, cast, overload

from tsdm.types.aliases import Nested
from tsdm.utils.funcutils import rpartial


@dataclass
class DecoratorError(Exception):
    r"""Raise Error related to decorator construction."""

    decorated: Fn
    r"""The decorator."""
    message: str = ""
    r"""Default message to print."""

    def __call__(self, *message_lines: str) -> Self:
        r"""Raise a new error."""
        # TODO: CHECK if dataclasses are the problem
        return DecoratorError(self.decorated, message="\n".join(message_lines))  # type: ignore[return-value]

    def __str__(self) -> str:
        r"""Create Error Message."""
        sign = signature(self.decorated)
        max_key_len = max(9, *(len(key) for key in sign.parameters))
        max_kind_len = max(len(str(param.kind)) for param in sign.parameters.values())
        default_message: tuple[str, ...] = (
            f"Signature: {sign}",
            "\n".join(
                f"{key.ljust(max_key_len)}: {str(param.kind).ljust(max_kind_len)}"
                f", Optional: {param.default is Parameter.empty}"
                for key, param in sign.parameters.items()
            ),
            self.message,
        )
        return super().__str__() + "\n" + "\n".join(default_message)


# region Protocol[Cls: type] -----------------------------------------------------------
# class BareClassDecorator[Cls: type](Protocol):
#     r"""Bare Decorator Protocol that preserves type."""
#
#     def __call__(self, cls: Cls, /) -> Cls: ...
#
#
# class ClassDecorator[Cls: type, **P](Protocol):
#     r"""Class Decorator Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, cls: Cls, /, *args: P.args, **kwargs: P.kwargs) -> Cls: ...
#     # fmt: on
#
#
# class ClassDecoratorFactory[Cls: type, **P](Protocol):
#     r"""Class Decorator Factory Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareClassDecorator[Cls]: ...
#     # fmt: on
#
#
# class ParametrizedClassDecorator[Cls: type, **P](Protocol):
#     r"""Parametrized Class Decorator Protocol that preserves type."""
#
#     # fmt: off
#     @overload  # @decorator(*args, **kwargs)
#     def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareClassDecorator[Cls]: ...
#     @overload  # @decorator
#     def __call__(self, cls: Cls, /, *args: P.args, **kwargs: P.kwargs) -> Cls: ...
#     # fmt: on


# endregion Protocol[Cls: type] --------------------------------------------------------


# region Protocol[type[T]] -------------------------------------------------------------
# class BareClassDecorator[T](Protocol):
#     r"""Bare Decorator Protocol that preserves type."""
#
#     def __call__(self, cls: type[T], /) -> type[T]: ...
#
#
# class ClassDecorator[T, **P](Protocol):
#     r"""Class Decorator Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, cls: type[T], /, *args: P.args, **kwargs: P.kwargs) -> type[T]: ...
#     # fmt: on
#
#
# class ClassDecoratorFactory[T, **P](Protocol):
#     r"""Class Decorator Factory Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareClassDecorator[T]: ...
#     # fmt: on
#
#
# class ParametrizedClassDecorator[T, **P](Protocol):
#     r"""Parametrized Class Decorator Protocol that preserves type."""
#
#     # fmt: off
#     @overload  # @decorator(*args, **kwargs)
#     def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareClassDecorator[T]: ...
#     @overload  # @decorator
#     def __call__(self, cls: type[T], /, *args: P.args, **kwargs: P.kwargs) -> type[T]: ...
#     # fmt: on


# endregion ----------------------------------------------------------------------------


# region Protocol[T] -------------------------------------------------------------------
# class BareClassDecorator[T](Protocol):
#     r"""Bare Decorator Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, cls: T, /) -> T: ...
#     # fmt: on
#
#
# class ClassDecorator[T, **P](Protocol):
#     r"""Class Decorator Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, cls: T, /, *args: P.args, **kwargs: P.kwargs) -> T: ...
#     # fmt: on
#
#
# class ClassDecoratorFactory[T, **P](Protocol):
#     r"""Class Decorator Factory Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareClassDecorator[T]: ...
#     # fmt: on
#
#
# class ParametrizedClassDecorator[T, **P](Protocol):
#     r"""Parametrized Class Decorator Protocol that preserves type."""
#
#     # fmt: off
#     @overload  # @decorator(*args, **kwargs)
#     def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareClassDecorator[T]: ...
#     @overload  # @decorator
#     def __call__(self, cls: T, /, *args: P.args, **kwargs: P.kwargs) -> T: ...
#     # fmt: on


# endregion Protocol[T] ----------------------------------------------------------------


# region ClassDecorator ----------------------------------------------------------------
class BareClassDecorator[Cls_in: type, Cls_out: type](Protocol):
    r"""Bare Function Decorator Protocol that preserves type."""

    def __call__(self, cls: Cls_in, /) -> Cls_out: ...


class ClassDecorator[Cls_in: type, Cls_out: type, **P](Protocol):  # -F_in, +F_out
    r"""Function Decorator Protocol that preserves type."""

    def __call__(
        self, cls: Cls_in, /, *args: P.args, **kwargs: P.kwargs
    ) -> Cls_out: ...


class ClassDecoratorFactory[Cls_in: type, Cls_out: type, **P](Protocol):
    r"""Function Decorator Factory Protocol that preserves type."""

    # fmt: off
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> ClassDecorator[Cls_in, Cls_out, P]: ...
    # fmt: on


class ParametrizedClassDecorator[Cls_in: type, Cls_out: type, **P](Protocol):
    r"""Parametrized Function Decorator Protocol that preserves type."""

    # fmt: off
    @overload  # @decorator(*args, **kwargs)
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareClassDecorator[Cls_in, Cls_out]: ...
    @overload  # @decorator
    def __call__(self, cls: Cls_in, /, *args: P.args, **kwargs: P.kwargs) -> Cls_out: ...
    # fmt: on


# endregion ClassDecorator -------------------------------------------------------------


# region FunctionDecorator -------------------------------------------------------------
class BareFunctionDecorator[F_in: Fn, F_out: Fn](Protocol):
    r"""Bare Function Decorator Protocol that preserves type."""

    def __call__(self, fn: F_in, /) -> F_out: ...


class FunctionDecorator[F_in: Fn, F_out: Fn, **P](Protocol):  # -F_in, +F_out
    r"""Function Decorator Protocol that preserves type."""

    def __call__(self, fn: F_in, /, *args: P.args, **kwargs: P.kwargs) -> F_out: ...


class FunctionDecoratorFactory[F_in: Fn, F_out: Fn, **P](Protocol):
    r"""Function Decorator Factory Protocol that preserves type."""

    # fmt: off
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> FunctionDecorator[F_in, F_out, P]: ...
    # fmt: on


class ParametrizedFunctionDecorator[F_in: Fn, F_out: Fn, **P](Protocol):
    r"""Parametrized Function Decorator Protocol that preserves type."""

    # fmt: off
    @overload  # @decorator
    def __call__(self, fn: F_in, /, *args: P.args, **kwargs: P.kwargs) -> F_out: ...
    @overload  # @decorator(*args, **kwargs)
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareFunctionDecorator[F_in, F_out]: ...
    # fmt: on


# endregion FunctionDecorator ----------------------------------------------------------


# region general decorators ------------------------------------------------------------
class BareDecorator[T_in, T_out](Protocol):
    r"""Protocol for decorators."""

    def __call__(self, obj: T_in, /) -> T_out: ...


class Decorator[T_in, T_out, **P](Protocol):
    r"""Protocol for decorators."""

    def __call__(self, obj: T_in, /, *args: P.args, **kwargs: P.kwargs) -> T_out: ...


class DecoratorFactory[T_in, T_out, **P](Protocol):
    r"""Protocol for parametrized decorators."""

    # fmt: off
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Decorator[T_in, T_out, P]: ...
    # fmt: on


class ParametrizedDecorator[T_in, T_out, **P](Protocol):
    r"""Protocol for parametrized decorators."""

    # fmt: off
    @overload  # @decorator(*args, **kwargs)
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> BareDecorator[T_in, T_out]: ...
    @overload  # @decorator
    def __call__(self, obj: T_in, /, *args: P.args, **kwargs: P.kwargs) -> T_out: ...
    # fmt: on


# endregion general decorators ---------------------------------------------------------

# class ParametrizedDecorator(Protocol[T, P]):
#     r"""Protocol for parametrized decorators."""
#
#     @overload
#     def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Decorator[T]: ...
#     @overload
#     def __call__(self, obj: T, /, *args: P.args, **kwargs: P.kwargs) -> T: ...


# NOTE: Type hinting is severely limited because HKTs are not supported.
#   The lack of an explicit FunctionType is a problem.
__SENTINEL = cast(Any, object())
r"""Sentinel object for distinguishing between BARE and FUNCTIONAL mode."""


# |                     |                      |                 | pyright | mypy |
# |---------------------|----------------------|-----------------|---------|------|
# | protocol[Cls: type] | decorator[Cls: type] | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | Y       | N    |
# |                     | decorator[type[T]]   | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | O       | A    |
# |                     | decorator[T]         | pprint[Cls]     | O       | N    |
# |                     |                      | pprint[type[T]] | O       | N    |
# | protocol[type[T]]   | decorator[Cls: type] | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | Y       | N    |
# |                     | decorator[type[T]]   | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | O       | N    |
# |                     | decorator[T]         | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | O       | A    |
# | protocol[T]         | decorator[Cls: type] | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | Y       | N    |
# |                     | decorator[type[T]]   | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | O       | A    |
# |                     | decorator[T]         | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | Y       | N    |


# |                     |                      |                 | pyright | mypy |
# |---------------------|----------------------|-----------------|---------|------|
# | protocol[T]         | decorator[Cls: type] | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | Y       | N    |
# |                     | decorator[type[T]]   | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | O       | A    |
# |                     | decorator[T]         | pprint[Cls]     | Y       | N    |
# |                     |                      | pprint[type[T]] | Y       | N    |


# FIXME: too complicated!
# REF: Code sample in [pyright playground](https://pyright-play.net/?enableExperimentalFeatures=true&code=GYJw9gtgBALgngBwJYDsDmUkQWEMoAK4MYAxmADYA0UYAbgKYgVgCGAJjawM7dMwB9eAgY0QDRqwpDEDAFChIUchQoNSMJGBTcAdKwBGpTNlz4AygwCOAVwYpS8uaQo9uUAEKtxAYVe8AEXVcVhIQAG0-bgAuWFkAXQAKIjASFQBKaLkoHKh2BmAoAQFSKWkBRL4KYBoXGKgomgB6dKgAWgA%2BBop63T65Z393Pzcg8hBQ3Eie2OFRKAAqBYIklLTKTOzc-MLi0tViyoZq2pnu7maaBe80eoJ9EFurhYBrAHcbu913z9bO89ifV0AxcbkI3lYEAYMBASAAXgx2CNAsEJmFpvU5s8VsliGQNllclAAMTKVgoKAGBhk1SIqAMJAwAAWTCgPCgAAF8uREksfo9uK1cGz3FzgsCiRz6EwWBwtjkdkUSmVDlUalAmldPrF7p9nvzbjrvh8BX8ul5fEMxiF0VF4oD%2BpLpcw2Ox5XkCkr9uUjidlGdGhqtQKjXrFq8TYbCMbfu0ulEHcC5IruTapgAVbFJVOxZHca1oqZzcLp%2BI0FbNM3giZQmHwxF5guTCLF0vl%2B1QIEDMXjZvJz0IBCwlCCACCJdmsnClls9kc8SSdVimY1VfTsXEMBsIApdW7qcLIH7hUHw8EHmnsRndgcDEXZ2L5jLq7jcRE047m%2B3u56A3EkmkOZElPVAx3SOR-wYKQZBEYCh1AgQPHAuQORAkcBFHQYwQAMTAMBEmvOcGEyTt%2BlQ%2BD0I8LDeE8bwCOsG9HBIrsIIkKDANkRJcLAVpSUfBiiPCABVFAXhQMA3hQBdWIAmCGESC1eLfBhwgteI5Dcfg5K4vCaGLbj4nAzS8G0i09KnNTwKAA)
# fmt: off
# @overload  # class-decorator
# def decorator[Cls: type, **P](deco: ClassDecorator[Cls, P], /) -> ParametrizedClassDecorator[Cls, P]: ...
# def decorator[T, **P](deco: ClassDecorator[type[T], P], /) -> ParametrizedClassDecorator[type[T], P]: ...
# def decorator[T, **P](deco: ClassDecorator[T, P], /) -> ParametrizedClassDecorator[T, P]: ...
@overload  # class decoration
def decorator[Cls_in: type, Cls_out: type, **P](
        deco: ClassDecorator[Cls_in, Cls_out, P], /
) -> ParametrizedClassDecorator[Cls_in, Cls_out, P]: ...
# def decorator[T_in, T_out, **P](deco: Decorator[type[T_in], type[T_out], P], /) -> ParametrizedDecorator[type[T_in], type[T_out], P]: ...
# def decorator[T_in, T_out, **P](deco: Decorator[T_in, T_out, P], /) -> ParametrizedDecorator[T_in, T_out, P]: ...
@overload  # function-decorator
def decorator[F_in: Fn, F_out: Fn, **P](
    deco: FunctionDecorator[F_in, F_out, P], /
) -> ParametrizedFunctionDecorator[F_in, F_out, P]: ...
# fmt: on
def decorator(deco, /):  # pyright: ignore[reportInconsistentOverload]
    r"""Meta-Decorator for constructing parametrized decorators.

    There are 3 different ways of using decorators:

    1. BARE MODE::

        @deco
        def func(*args, **kwargs):
            # Input: func
            # Output: Wrapped Function

    2. FUNCTIONAL MODE::

        deco(func, *args, **kwargs)
        # Input: func, args, kwargs
        # Output: Wrapped Function

    3. BRACKET MODE::

        @deco(*args, **kwargs)
        def func(*args, **kwargs):
            # Input: args, kwargs
            # Output: decorator with single positional argument

    In order to distinguish between these modes, we require that the decorator has
    the following signature::

        def deco(obj, /, *, ...):

    That is, it must hat exactly one positional argument: the object to be decorated.
    """
    logger = logging.getLogger(f"@decorator/{deco.__name__}")
    logger.debug("Creating @decorator.")

    deco_sig = signature(deco)
    ErrorHandler = DecoratorError(deco)

    for param in deco_sig.parameters.values():
        has_default = param.default is not Parameter.empty

        match param.kind, has_default:
            case Parameter.POSITIONAL_ONLY, True:
                raise ErrorHandler(
                    "@decorator does not support POSITIONAL_ONLY arguments with defaults!"
                )
            case Parameter.POSITIONAL_OR_KEYWORD, _:
                raise ErrorHandler(
                    "@decorator does not support POSITIONAL_OR_KEYWORD arguments!",
                    "Separate positional and keyword arguments using '/' and '*':"
                    ">>> def deco(func, /, *, ko1, ko2, **kwargs): ...",
                    "See https://www.python.org/dev/peps/pep-0570/",
                )
            case Parameter.VAR_POSITIONAL, _:
                raise ErrorHandler(
                    "@decorator does not support VAR_POSITIONAL arguments!",
                )

    @wraps(deco)
    def __parametrized_decorator(obj=__SENTINEL, /, *args, **kwargs):
        if obj is __SENTINEL:
            logger.debug(
                "@decorator used in BRACKET mode.\n"
                "Creating decorator with fixed arguments \n\targs=%s, \n\tkwargs=%s",
                args,
                kwargs,
            )
            return rpartial(deco, *args, **kwargs)

        logger.debug("@decorator used in FUNCTIONAL/BARE mode.")
        return deco(obj, *args, **kwargs)

    return __parametrized_decorator


def attribute[T, R](func: Fn[[T], R], /) -> R:  # T, +R
    r"""Create a decorator that converts method to attribute."""

    @wraps(func, updated=())
    class __attribute:
        __slots__ = ("func", "payload")
        __SENTINEL = object()
        func: Fn[[T], R]
        payload: R

        def __init__(self, function: Fn) -> None:
            self.func = function
            self.payload = cast(Any, self.__SENTINEL)

        @overload
        def __get__(self, obj: None, obj_type: Optional[type] = ...) -> Self: ...
        @overload
        def __get__(self, obj: T, obj_type: Optional[type] = ...) -> R: ...
        def __get__(self, obj, obj_type=None):
            if obj is None:
                return self
            if self.payload is self.__SENTINEL:
                self.payload = self.func(obj)
            return self.payload

    return cast(R, __attribute(func))


def recurse_on_container[T, R](  # T, +R
    func: Fn[[T], R], /, *, leaf_type: type[T]
) -> Fn[[Nested[T]], Nested[R]]:
    r"""Apply function to a nested iterables of a given kind.

    Args:
        leaf_type: The type of the leave nodes
        func: A function to apply to all leave Nodes
    """

    @wraps(func)
    def recurse(x: Nested[T]) -> Nested[R]:
        match x:
            case leaf_type():  # type: ignore[misc]
                return func(x)  # type: ignore[unreachable]
            case dict(mapping):
                return {k: recurse(v) for k, v in mapping.items()}
            case list(seq):
                return [recurse(obj) for obj in seq]
            case tuple(seq):
                return tuple(recurse(obj) for obj in seq)
            case set(items):
                return {recurse(obj) for obj in items}  # pyright: ignore[reportUnhashable]
            case frozenset(items):
                return frozenset(recurse(obj) for obj in items)
            case _:
                raise TypeError(f"Unsupported type: {type(x)}")

    return recurse
