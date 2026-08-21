r"""Submodule containing general purpose decorators."""

__all__ = [
    # Classes
    "Decorator",
    "DecoratorError",
    "ParametrizedDecorator",
    "ParametrizedPolymorphism",
    # Functions
    "decorator",
    "rpartial",
]

import logging
from collections.abc import Callable as Fn
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Protocol, Self, cast, overload

from .types.callbacks import Polymorphism


def rpartial[**P, R](  # +R
    func: Fn[P, R], /, *fixed_args: Any, **fixed_kwargs: Any
) -> Fn[..., R]:
    r"""Apply positional arguments from the right.

    References:
        - https://docs.python.org/3/library/functools.html#functools.partial
        - https://github.com/python/typeshed/blob/bbd9dd1c4f596f564542d48bb05b2cc2e2a7a28d/stdlib/functools.pyi#L129
    """

    @wraps(func)
    def __wrapper(*func_args: Any, **func_kwargs: Any) -> R:
        # FIXME: https://github.com/python/typeshed/issues/8703
        return func(*(func_args + fixed_args), **(func_kwargs | fixed_kwargs))

    return __wrapper


@dataclass(slots=True, frozen=True)
class DecoratorError(Exception):
    r"""Raise Error related to decorator construction."""

    decorated: Fn
    r"""The decorator."""
    message: str = ""
    r"""Default message to print."""

    def __call__(self, *message_lines: str) -> Self:
        r"""Raise a new error."""
        # TODO: CHECK if dataclasses are the problem
        return self.__class__(self.decorated, message="\n".join(message_lines))

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


class Decorator[T_in, T_out, **P](Protocol):
    r"""Protocol for decorators."""

    def __call__(self, obj: T_in, /, *args: P.args, **kwargs: P.kwargs) -> T_out: ...


class ParametrizedDecorator[T_in, T_out, **P](Protocol):
    r"""Protocol for parametrized decorators."""

    # shared attributes with classes `type` and `function`
    __name__: str
    __module__: str
    __qualname__: str
    __annotations__: dict[str, Any]

    # fmt: off
    @overload  # @decorator / decorator(obj, *args, **kwargs)
    def __call__(self, obj: T_in, /, *arg: P.args, **kwargs: P.kwargs) -> T_out: ...
    @overload  # @decorator(*args, **kwargs)
    def __call__(self, /, *arg: P.args, **kwargs: P.kwargs) -> Fn[[T_in], T_out]: ...
    # fmt: on


class ParametrizedPolymorphism[**P](Protocol):
    r"""Polymorphic Decorator Protocol."""

    # shared attributes with classes `type` and `function`
    __name__: str
    __module__: str
    __qualname__: str
    __annotations__: dict[str, Any]

    # fmt: off
    @overload  # @decorator / decorator(obj, *args, **kwargs)
    def __call__[T: Any](self, obj: T, /, *arg: P.args, **kwargs: P.kwargs) -> T: ...
    @overload  # @decorator(*args, **kwargs)
    def __call__(self, /, *arg: P.args, **kwargs: P.kwargs) -> Polymorphism: ...
    # fmt: on


_OBJ = cast("Any", object())
r"""Sentinel object for distinguishing between BARE and FUNCTIONAL mode."""


# FIXME: https://github.com/facebook/pyrefly/issues/4461
# @overload
# def decorator[X, Y, **P](deco: Decorator[X, Y, P], /) -> ParametrizedDecorator[X, Y, P]: ...
def decorator[**P](deco: Polymorphism[P], /) -> ParametrizedPolymorphism[P]:
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
    logger = logging.getLogger(f"@decorator/{deco.__name__}")  # type: ignore
    logger.debug("Creating @decorator.")

    deco_sig = signature(deco)
    ErrorHandler = DecoratorError(deco)

    for param in deco_sig.parameters.values():
        match param.kind:
            case Parameter.POSITIONAL_ONLY:
                if param.default is not Parameter.empty:
                    raise ErrorHandler(
                        "@decorator does not support POSITIONAL_ONLY arguments with defaults!"
                    )
            case Parameter.POSITIONAL_OR_KEYWORD:
                raise ErrorHandler(
                    "@decorator does not support POSITIONAL_OR_KEYWORD arguments!",
                    "Separate positional and keyword arguments using '/' and '*':",
                    ">>> def deco(func, /, *, ko1, ko2, **kwargs): ...",
                    "See https://www.python.org/dev/peps/pep-0570/",
                )
            case Parameter.VAR_POSITIONAL:
                raise ErrorHandler(
                    "@decorator does not support VAR_POSITIONAL arguments!",
                )
            case Parameter.VAR_KEYWORD | Parameter.KEYWORD_ONLY:
                pass

    @wraps(deco)
    def _deco(obj=_OBJ, /, *args, **kwargs):
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

    return _deco  # pyright: ignore[reportReturnType]
