r"""Submodule containing general purpose decorators."""

__all__ = [
    # Classes
    "Polymorphism",
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
from typing import Any, Protocol, Self, overload


class Polymorphism[**P = []](Protocol):
    r"""Protocol for callables that preserve the type of their first argument."""

    def __call__[T: Any](self, obj: T, /, *arg: P.args, **kwargs: P.kwargs) -> T: ...


def rpartial[**P, R](  # +R
    func: Fn[P, R], /, *fixed_args: Any, **fixed_kwargs: Any
) -> Fn[..., R]:
    r"""Apply positional arguments from the right.

    Return a wrapper that appends `fixed_args` to its positional arguments.
    Fixed keyword arguments override keyword arguments supplied to the wrapper.

    References:
        - [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial).
        - [Typeshed's partial annotations](https://github.com/python/typeshed/blob/bbd9dd1c4f596f564542d48bb05b2cc2e2a7a28d/stdlib/functools.pyi#L129).
    """

    @wraps(func)
    def __wrapper(*func_args: Any, **func_kwargs: Any) -> R:
        # FIXME: https://github.com/python/typeshed/issues/8703
        return func(*(func_args + fixed_args), **(func_kwargs | fixed_kwargs))

    return __wrapper


@dataclass(slots=True, frozen=True)
class DecoratorError(Exception):
    r"""Exception describing an invalid decorator signature."""

    decorated: Fn
    r"""The decorator."""
    message: str = ""
    r"""Additional explanation included in the formatted error."""

    def __call__(self, *message_lines: str) -> Self:
        r"""Return a new exception with the supplied message lines joined by newlines."""
        # TODO: CHECK if dataclasses are the problem
        return self.__class__(self.decorated, message="\n".join(message_lines))

    def __str__(self) -> str:
        r"""Format the error with the decorator's signature and parameter details."""
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


_MISSING: Any = sentinel("_MISSING")
r"""Sentinel distinguishing a decorator factory call from direct decoration."""


# FIXME: https://github.com/facebook/pyrefly/issues/4461
# @overload
# def decorator[X, Y, **P](deco: Decorator[X, Y, P], /) -> ParametrizedDecorator[X, Y, P]: ...
def decorator[**P](deco: Polymorphism[P], /) -> ParametrizedPolymorphism[P]:
    r"""Construct a decorator supporting direct calls and decorator factory calls.

    The returned decorator supports three forms:

    1. Bare decoration:

        ```python
        @deco
        def func(*args, **kwargs): ...
        ```

    2. A direct function call:

        ```python
        deco(func, **kwargs)
        ```

    3. A decorator factory call:

        ```python
        @deco(**kwargs)
        def func(*args, **kwargs): ...
        ```

    The intended signature has exactly one required positional-only parameter
    for the object to decorate, followed by keyword-only options:

    ```python
    def deco(obj, /, *, option=None): ...
    ```

    The signature check rejects positional-only parameters with defaults,
    positional-or-keyword parameters, and variadic positional parameters.

    Args:
        deco: Decorator implementation to wrap.

    Returns:
        A wrapper supporting the three calling forms.

    Raises:
        DecoratorError: If the signature contains an unsupported parameter.
    """
    name = getattr(deco, "__name__", "<unknown>")
    logger = logging.getLogger(f"@decorator/{name}")
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
    def _deco(obj=_MISSING, /, *args, **kwargs):
        if obj is _MISSING:
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
