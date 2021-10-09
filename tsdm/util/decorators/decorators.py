r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Functions
    "decorator",
    "sphinx_value",
    "timefun",
    # Exceptions
    "DecoratorError",
]

import gc
import logging
import os
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature
from time import perf_counter_ns
from typing import Any, Callable, Optional

LOGGER = logging.getLogger(__name__)

KEYWORD_ONLY = Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = Parameter.VAR_KEYWORD
VAR_POSITIONAL = Parameter.VAR_POSITIONAL
EMPTY = Parameter.empty


def rpartial(func: Callable, /, *fixed_args: Any, **fixed_kwargs: Any) -> Callable:
    r"""Apply positional arguments from the right."""

    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        return func(*(func_args + fixed_args), **(func_kwargs | fixed_kwargs))

    return wrapper


@dataclass
class DecoratorError(Exception):
    r"""Raise Error related to decorator construction."""

    decorated: Callable
    r"""The decorator."""
    message: str = ""
    r"""Default message to print."""

    def __call__(self, *message_lines):
        r"""Raise a new error."""
        return DecoratorError(self.decorated, message="\n".join(message_lines))

    def __str__(self):
        r"""Create Error Message."""
        sign = signature(self.decorated)
        maxkey = max(9, max(len(key) for key in sign.parameters))
        maxkind = max(len(str(param.kind)) for param in sign.parameters.values())
        default_message: tuple[str, ...] = (
            f"Signature: {sign}",
            "\n".join(
                f"{key.ljust(maxkey)}: {str(param.kind).ljust(maxkind)}"
                f", Optional: {param.default is EMPTY}"
                for key, param in sign.parameters.items()
            ),
            self.message,
        )
        return super().__str__() + "\n" + "\n".join(default_message)


def decorator(deco: Callable) -> Callable:
    r"""Meta-Decorator for constructing parametrized decorators."""
    ErrorHandler = DecoratorError(deco)
    mandatory_pos_args, mandatory_key_args = set(), set()

    for key, param in signature(deco).parameters.items():
        if param.kind is VAR_POSITIONAL:
            # TODO: allow VAR_POSITIONAL, iff mandatory KW_ONLY present.
            raise ErrorHandler("VAR_POSITIONAL arguments (*args) not allowed!")
        if param.kind is POSITIONAL_OR_KEYWORD:
            # TODO: allow VAR_POSITIONAL, iff mandatory KW_ONLY present.
            raise ErrorHandler(
                "Decorator does not support POSITIONAL_OR_KEYWORD arguments!!",
                "Separate positional and keyword arguments: fun(po, /, *, ko=None,)",
                "Cf. https://www.python.org/dev/peps/pep-0570/",
            )
        if param.kind is POSITIONAL_ONLY and param.default is not EMPTY:
            raise ErrorHandler("POSITIONAL_ONLY arguments not allowed to be optional!")
        if param.default is EMPTY and param.kind is POSITIONAL_ONLY:
            mandatory_pos_args |= {key}
        if param.default is EMPTY and param.kind is KEYWORD_ONLY:
            mandatory_key_args |= {key}

    if not mandatory_pos_args:
        raise ErrorHandler(
            "First argument of decorator must be POSITIONAL_ONLY (the function to be wrapped)!"
        )

    @wraps(deco)
    def parametrized_decorator(  # pylint: disable=keyword-arg-before-vararg
        __func__: Any = None, *args: Any, **kwargs: Any
    ) -> Callable:
        if len(mandatory_pos_args | mandatory_key_args) > 1:
            # no bare decorator allowed!
            if len(args) + 1 == len(mandatory_pos_args) - 1:
                # all pos args except func given
                if missing_keys := (mandatory_key_args - kwargs.keys()):
                    raise ErrorHandler(f"Not enough kwargs supplied, {missing_keys=}")
                LOGGER.info(">>> Generating bracket version of %s <<<", decorator)
                return rpartial(deco, *(__func__, *args), **kwargs)
            LOGGER.info(">>> Generating functional version of %s <<<", decorator)
            return deco(__func__, *args, **kwargs)
        if __func__ is None:
            LOGGER.info(">>> Generating bare version of %s <<<", decorator)
            return rpartial(deco, *args, **kwargs)
        LOGGER.info(">>> Generating bracket version of %s <<<", decorator)
        return deco(__func__, *args, **kwargs)

    return parametrized_decorator


@decorator
def timefun(
    fun: Callable, /, *, append: bool = True, loglevel: int = logging.WARNING
) -> Callable:
    r"""Log the execution time of the function. Use as decorator.

    By default appends the execution time (in seconds) to the function call.

    ``outputs, time_elapse = timefun(f, append=True)(inputs)``

    If the function call failed, ``outputs=None`` and ``time_elapsed=float('nan')`` are returned.

    Parameters
    ----------
    fun: Callable
    append: bool, default=True
        Whether to append the time result to the function call
    loglevel: int, default=logging.Warning (20)
    """
    timefun_logger = logging.getLogger("timefun")

    @wraps(fun)
    def timed_fun(*args, **kwargs):
        gc.collect()
        gc.disable()
        try:
            start_time = perf_counter_ns()
            result = fun(*args, **kwargs)
            end_time = perf_counter_ns()
            elapsed = (end_time - start_time) / 10 ** 9
            timefun_logger.log(loglevel, "%s executed in %.4f s", fun.__name__, elapsed)
        except (KeyboardInterrupt, SystemExit) as E:
            raise E
        except Exception as E:  # pylint: disable=W0703
            result = None
            elapsed = float("nan")
            RuntimeWarning(f"Function execution failed with Exception {E}")
            timefun_logger.log(loglevel, "%s failed with Exception %s", fun.__name__, E)
        gc.enable()

        return (result, elapsed) if append else result

    return timed_fun


@decorator
def sphinx_value(func: Callable, value: Any, /) -> Callable:
    """Use alternative attribute value during sphinx compilation - useful for attributes.

    Parameters
    ----------
    func: Callable
    value: Any

    Returns
    -------
    Callable
    """

    @wraps(func)
    def wrapper(*func_args, **func_kwargs):  # pylint: disable=unused-argument
        return value

    return wrapper if os.environ.get("GENERATING_DOCS", False) else func


# TODO: implement mutually_exclusive_args wrapper
# idea: sometimes we have a tuple of args (a ,b ,c, ...) where
# 1. at least x of these args are required
# 2. at most y of these args are allowed.
# this decorator raises and error if not the right number of args is supplied.
# alternative:
# - supply a single int: exact allowed number of args, e.g. [0, 1, 3] if
# exactly 0, 1 or 3 arguments allowed.
# - supply a tuple[int, int]: min/max number of allowed args
# - supply a list[int] of allowed number of args, e.g.
# - supply a list[tuple[str]] of allowed combinations, e.g. [("a", "b"), ("c",), ("a', "c"), ...]
# Union[int, tuple[int, int], list[int], list[tuple[str, ...]]]

# def exclusive_args(args: tuple[str, ...],
# allowed: Union[int, tuple[int, int], list[int], list[tuple[str, ...]]]):
#     pass
