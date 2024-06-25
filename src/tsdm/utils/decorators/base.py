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
    # Functions
    "attribute",
    "decorator",
    "recurse_on_container",
]

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature

from typing_extensions import Any, Optional, ParamSpec, Protocol, Self, cast, overload

from tsdm.types.aliases import Nested
from tsdm.types.variables import F_co, F_contra, P, R_co, T
from tsdm.utils.funcutils import rpartial

D = ParamSpec("D")


@dataclass
class DecoratorError(Exception):
    r"""Raise Error related to decorator construction."""

    decorated: Callable
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


class ClassDecorator(Protocol[P]):
    r"""Class Decorator Protocol that preserves type."""

    # fmt: off
    def __call__(self, cls: type[T], /, *args: P.args, **kwargs: P.kwargs) -> type[T]: ...
    # fmt: on


class ClassDecoratorFactory(Protocol[P]):
    r"""Class Decorator Factory Protocol that preserves type."""

    # fmt: off
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> ClassDecorator[P]: ...
    # fmt: on


class ParametrizedClassDecorator(Protocol[P]):
    r"""Parametrized Class Decorator Protocol that preserves type."""

    # fmt: off
    @overload
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> ClassDecorator[P]: ...
    @overload
    def __call__(self, cls: type[T], /, *args: P.args, **kwargs: P.kwargs) -> type[T]: ...
    # fmt: on


# class FunctionDecorator(Protocol[D, P, T_contra, P2, T2_co]):
#     r"""Function Decorator Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, func: Callable[P, T_contra], /, *args: D.args, **kwargs: D.kwargs) -> Callable[P2, T2_co]: ...
#     # fmt: on
#
#
# class FunctionDecoratorFactory(Protocol[D, P, T_contra, P2, T2_co]):
#     r"""Function Decorator Factory Protocol that preserves type."""
#
#     # fmt: off
#     def __call__(self, /, *args: D.args, **kwargs: D.kwargs) -> FunctionDecorator[D, P, T_contra, P2, T2_co]: ...
#     # fmt: on
#
#
# class ParametrizedFunctionDecorator(Protocol[D, P, T_contra, P2, T2_co]):
#     r"""Parametrized Function Decorator Protocol that preserves type."""
#
#     # fmt: off
#     @overload
#     def __call__(self, /, *args: D.args, **kwargs: D.kwargs) -> FunctionDecorator[D, P, T_contra, P2, T2_co]: ...
#     @overload
#     def __call__(self, func: Callable[P, T_contra], /, *args: D.args, **kwargs: D.kwargs) -> Callable[P2, T2_co]: ...
#     # fmt: on


class FunctionDecorator(Protocol[P, F_contra, F_co]):
    r"""Function Decorator Protocol that preserves type."""

    # fmt: off
    def __call__(self, func: F_contra, /, *args: P.args, **kwargs: P.kwargs) -> F_co: ...
    # fmt: on


class FunctionDecoratorFactory(Protocol[P, F_contra, F_co]):
    r"""Function Decorator Factory Protocol that preserves type."""

    # fmt: off
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> FunctionDecorator[P, F_contra, F_co]: ...
    # fmt: on


class ParametrizedFunctionDecorator(Protocol[P, F_contra, F_co]):
    r"""Parametrized Function Decorator Protocol that preserves type."""

    # fmt: off
    @overload
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> FunctionDecorator[P, F_contra, F_co]: ...
    @overload
    def __call__(self, func: F_contra, /, *args: P.args, **kwargs: P.kwargs) -> F_co: ...
    # fmt: on


class Decorator(Protocol[P]):
    r"""Protocol for decorators."""

    def __call__(self, obj: T, /, *args: P.args, **kwargs: P.kwargs) -> T: ...


class DecoratorFactory(Protocol[P]):
    r"""Protocol for parametrized decorators."""

    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Decorator: ...


class ParametrizedDecorator(Protocol[P]):
    r"""Protocol for parametrized decorators."""

    @overload
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Decorator: ...
    @overload
    def __call__(self, obj: T, /, *args: P.args, **kwargs: P.kwargs) -> T: ...


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


# fmt: off
# FIXME: too complicated!
@overload  # class-decorator
def decorator(deco: ClassDecorator[P], /) -> ParametrizedClassDecorator[P]: ...
@overload  # function-decorator
def decorator(deco: FunctionDecorator[P, F_contra, F_co], /) -> ParametrizedFunctionDecorator[P, F_contra, F_co]: ...
def decorator(deco, /):  # pyright: ignore[reportInconsistentOverload]
    # fmt: on
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


def attribute(func: Callable[[T], R_co], /) -> R_co:
    r"""Create a decorator that converts method to attribute."""

    @wraps(func, updated=())
    class __attribute:
        __slots__ = ("func", "payload")
        sentinel = object()
        func: Callable[[T], R_co]
        payload: R_co

        def __init__(self, function: Callable) -> None:
            self.func = function
            self.payload = cast(Any, self.sentinel)

        def __get__(
            self, obj: T | None, obj_type: Optional[type] = None
        ) -> Self | R_co:
            if obj is None:
                return self
            if self.payload is self.sentinel:
                self.payload = self.func(obj)
            return self.payload

    return cast(R_co, __attribute(func))


def recurse_on_container(
    func: Callable[[T], R_co], /, *, leaf_type: type[T]
) -> Callable[[Nested[T]], Nested[R_co]]:
    r"""Apply function to a nested iterables of a given kind.

    Args:
        leaf_type: The type of the leave nodes
        func: A function to apply to all leave Nodes
    """

    @wraps(func)
    def recurse(x: Nested[T]) -> Nested[R_co]:
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


def _extends(parent_func: Callable[P, None], /) -> Callable[[Callable], Callable]:
    r"""Decorator to extend a parent function.

    For example, when one wants to extend the __init__ of a parent class
    with an extra argument.

    This will synthesize a new function that combines the extra arguments with
    the ones of the parent function. The new arguments passed to the synthesized
    function are available within the function body.

    Example:
        class Parent:
            def foo(self, a, b, /, *, key):
                ...

        class Child(Parent):
            @extends(Parent.foo)
            def foo(self, c, *parent_args, *, bar, **parent_kwargs):
                super().foo(*parent_args, **parent_kwargs)
                ...

    the synthesized function will roughly look like this:

        def __synthetic__init__(self, a, b, c, /, *, foo, bar):
            parent_args = (a, b)
            parent_kwargs = dict(key=key)
            func_args = (c,)
            func_kwargs = dict(bar=bar)
            wrapped_func(*parent_args, *func_args, **parent_kwargs, **func_kwargs)

    Note:
        - neither parent nor child func may have varargs.
        - The child func may reuse keyword arguments from the parent func and give them different keywords.
          - if keyword args are reused, they won't be included in parent_kwargs.
        - additional positional only args must have defaults values (LSP!)
        - additional positional only arguments are always added after positional-only arguments of the parent.
    """
    raise NotImplementedError(f"Not yet implemented. {parent_func=}")
