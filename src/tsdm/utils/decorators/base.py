r"""Submodule containing general purpose decorators."""

__all__ = [
    # Classes
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
    "PolymorphicDecorator",
    "PolymorphicClassDecorator",
    "PolymorphicFunctionDecorator",
    # Functions
    "decorator",
    "recurse_on_container",
]

import logging
from collections.abc import Callable as Fn
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Protocol, Self, cast, overload

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


# region ClassDecorator ----------------------------------------------------------------
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
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[Cls_in], Cls_out]: ...
    @overload  # @decorator
    def __call__(self, cls: Cls_in, /, *args: P.args, **kwargs: P.kwargs) -> Cls_out: ...
    # fmt: on


class PolymorphicClassDecorator[**P](Protocol):
    r"""Polymorphic Class Decorator Protocol."""

    # fmt: off
    @overload  # @decorator
    def __call__[Cls: type](self, cls: Cls, /, *args: P.args, **kwargs: P.kwargs) -> Cls: ...
    @overload  # @decorator(*args, **kwargs)
    def __call__[Cls: type](self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[Cls], Cls]: ...
    # fmt: on


# endregion ClassDecorator -------------------------------------------------------------


# region FunctionDecorator -------------------------------------------------------------
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
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[F_in], F_out]: ...
    # fmt: on


class PolymorphicFunctionDecorator[**P](Protocol):
    r"""Polymorphic Function Decorator Protocol."""

    # fmt: off
    @overload  # @decorator
    def __call__[F: Fn](self, fn: F, /, *args: P.args, **kwargs: P.kwargs) -> F: ...
    @overload  # @decorator(*args, **kwargs)
    def __call__[F: Fn](self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[F], F]: ...
    # fmt: on


# endregion FunctionDecorator ----------------------------------------------------------


# region general decorators ------------------------------------------------------------
class Decorator[T_in, T_out, **P](Protocol):
    r"""Protocol for decorators."""

    # shared attributes with classes `type` and `function`
    __name__: str
    __module__: str
    __qualname__: str
    __annotations__: dict[str, Any]

    def __call__(self, obj: T_in, /, *args: P.args, **kwargs: P.kwargs) -> T_out: ...


class DecoratorFactory[T_in, T_out, **P](Protocol):
    r"""Protocol for parametrized decorators."""

    # fmt: off
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Decorator[T_in, T_out, P]: ...
    # fmt: on


class ParametrizedDecorator[T_in, T_out, **P](Protocol):
    r"""Protocol for parametrized decorators."""

    # shared attributes with classes `type` and `function`
    __name__: str
    __module__: str
    __qualname__: str
    __annotations__: dict[str, Any]

    # fmt: off
    @overload  # @decorator(*args, **kwargs)
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[T_in], T_out]: ...
    @overload  # @decorator / decorator(obj, *args, **kwargs)
    def __call__(self, obj: T_in, /, *args: P.args, **kwargs: P.kwargs) -> T_out: ...
    # fmt: on


class PolymorphicDecorator[**P](Protocol):
    r"""Polymorphic Decorator Protocol."""

    # fmt: off
    @overload  # @decorator
    def __call__[T](self, obj: T, /, *args: P.args, **kwargs: P.kwargs) -> T: ...
    @overload  # @decorator(*args, **kwargs)
    def __call__[T](self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[T], T]: ...
    # fmt: on


# endregion general decorators ---------------------------------------------------------

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

# FIXME: https://github.com/python/mypy/issues/17191
#   Somehow broken in mypy...
# NOTE: Type hinting is severely limited because HKTs are not supported.
#   The lack of an explicit FunctionType is a problem.
# @overload  # class-decorator
# def decorator[Cls: type, **P](
#     deco: ClassDecorator[Cls, P], /
# ) -> ParametrizedClassDecorator[Cls, P]: ...
# def decorator[T, **P](
#     deco: ClassDecorator[type[T], P], /
# ) -> ParametrizedClassDecorator[type[T], P]: ...
# def decorator[T, **P](
#     deco: ClassDecorator[T, P], /
# ) -> ParametrizedClassDecorator[T, P]: ...
# @overload  # class decoration
# def decorator[Cls_in: type, Cls_out: type, **P](
#     deco: ClassDecorator[Cls_in, Cls_out, P], /
# ) -> ParametrizedClassDecorator[Cls_in, Cls_out, P]: ...
# def decorator[T_in, T_out, **P](
#     deco: Decorator[type[T_in], type[T_out], P], /
# ) -> ParametrizedDecorator[type[T_in], type[T_out], P]: ...
# def decorator[T_in, T_out, **P](
#     deco: Decorator[T_in, T_out, P], /
# ) -> ParametrizedDecorator[T_in, T_out, P]: ...
# @overload  # function-decorator
# def decorator[F_in: Fn, F_out: Fn, **P](
#     deco: FunctionDecorator[F_in, F_out, P], /
# ) -> ParametrizedFunctionDecorator[F_in, F_out, P]: ...
# def decorator(deco, /):

_OBJ = cast(Any, object())
r"""Sentinel object for distinguishing between BARE and FUNCTIONAL mode."""


def decorator[X, Y, **P](deco: Decorator[X, Y, P], /) -> ParametrizedDecorator[X, Y, P]:
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

    # FIXME: Instead of inner function, return instance of ParametrizedDecorator
    @overload  # @decorator(*args, **kwargs)
    def _deco(*args: P.args, **kwargs: P.kwargs) -> Fn[[X], Y]: ...  # type: ignore[overload-overlap]
    @overload  # @decorator / decorator(obj, *args, **kwargs)
    def _deco(obj: X, /, *args: P.args, **kwargs: P.kwargs) -> Y: ...
    @wraps(deco)  # type: ignore[misc]
    def _deco(obj: X = _OBJ, /, *args: P.args, **kwargs: P.kwargs) -> Y | Fn[[X], Y]:
        if obj is _OBJ:
            logger.debug(
                "@decorator used in BRACKET mode.\n"
                "Creating decorator with fixed arguments \n\targs=%s, \n\tkwargs=%s",
                args,
                kwargs,
            )
            return rpartial(deco, *args, **kwargs)

        logger.debug("@decorator used in FUNCTIONAL/BARE mode.")
        return deco(obj, *args, **kwargs)

    return _deco


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
