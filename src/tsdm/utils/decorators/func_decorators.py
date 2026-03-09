r"""Function decorators for wrapping functions with additional functionality."""

__all__ = [
    # ABCs & Protocols
    # Functions
    "debug",
    "return_namedtuple",
    "timefun",
    "trace",
    "wrap_func",
    "wrap_method",
]

import gc
import logging
from collections.abc import Callable as Fn, Sequence
from functools import wraps
from time import perf_counter_ns
from typing import Concatenate, NamedTuple, Optional

from tsdm.constants import EMPTY_FN
from tsdm.types.namedtuple import NTuple
from tsdm.utils.decorators.base import DecoratorError, decorator
from tsdm.utils.funcutils import get_exit_point_names


# region without @decorator ------------------------------------------------------------
def debug[**P, R](func: Fn[P, R], /) -> Fn[P, R]:  # +R
    r"""Print the function signature and return value."""
    logger = logging.getLogger(f"debug@{func.__name__}")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args_repr = [f"{type(a)}" for a in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        passed_args = ", ".join(args_repr + kwargs_repr)
        logger.info("Calling with arguments %s", passed_args)
        value = func(*args, **kwargs)
        logger.info("Return value %s", value)
        return value

    return __wrapper


def trace[**P, R](func: Fn[P, R], /) -> Fn[P, R]:  # +R
    r"""Log entering and exiting of function."""
    logger = logging.getLogger("trace")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.info(
            "%s",
            "\n\t".join(
                (
                    f"{func.__qualname__}: ENTERING",
                    f"args={tuple(type(arg).__name__ for arg in args)}",
                    f"kwargs={ {k: type(v).__name__ for k, v in kwargs.items()}!s}",
                )
            ),
        )
        try:
            logger.info("%s: EXECUTING", func.__qualname__)
            result = func(*args, **kwargs)
        except Exception as exc:
            logger.exception("Execution of %s failed!", func.__qualname__)
            raise RuntimeError(
                f"Function execution failed with Exception {exc}"
            ) from exc
        logger.info(
            "%s: SUCCESS with result=%s", func.__qualname__, type(result).__name__
        )
        logger.info("%s", "\n\t".join((f"{func.__qualname__}: EXITING",)))
        return result

    return __wrapper


# endregion without @decorator ---------------------------------------------------------


@decorator
def timefun[**P, R](  # +R
    func: Fn[P, R], /, *, loglevel: int = logging.WARNING
) -> Fn[P, tuple[R, float]]:
    r"""Log the execution time of the function. Use as decorator.

    By default, appends the execution time (in seconds) to the function call.

    `outputs, time_elapse = timefun(f, append=True)(inputs)`

    If the function call failed, `outputs=None` and `time_elapsed=float('nan')` are returned.

    If `append=True`, then the decorated function will return a tuple of the form `(func(x), time_elapsed)`.
    """
    timefun_logger = logging.getLogger("timefun")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
        gc.collect()
        gc.disable()
        try:
            start_time = perf_counter_ns()
            result = func(*args, **kwargs)
            end_time = perf_counter_ns()
            elapsed = (end_time - start_time) / 10**9
            timefun_logger.log(
                loglevel, "%s executed in %.4f s", func.__qualname__, elapsed
            )
        except Exception as exc:
            timefun_logger.exception("Execution of %s failed!", func.__qualname__)
            raise RuntimeError("Function execution failed") from exc
        finally:
            gc.enable()

        return result, elapsed

    return __wrapper


@decorator
def wrap_func[**P, R](  # +R
    func: Fn[P, R],
    /,
    *,
    before: Optional[Fn[..., None]] = None,
    after: Optional[Fn[..., None]] = None,
    pass_args: bool = False,
) -> Fn[P, R]:
    r"""Wrap a function with pre- and post-hooks."""
    logger = logging.getLogger(f"{__name__}/{wrap_func.__name__}/{func.__name__}")
    logger.debug("Wrapping function")

    pre_func: Fn[..., None] = (
        EMPTY_FN
        if before is None
        else (before if pass_args else lambda *_, **__: before())
    )
    post_func: Fn[..., None] = (
        EMPTY_FN
        if after is None
        else (after if pass_args else lambda *_, **__: after())
    )

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        pre_func(*args, **kwargs)
        result = func(*args, **kwargs)
        post_func(*args, **kwargs)
        return result

    return __wrapper


@decorator
def wrap_method[**P, T, R](  # T, +R
    method: Fn[Concatenate[T, P], R],
    /,
    *,
    before: Optional[Fn[Concatenate[T, ...], None]] = None,
    after: Optional[Fn[Concatenate[T, ...], None]] = None,
    pass_args: bool = False,
) -> Fn[Concatenate[T, P], R]:
    r"""Wrap a function with pre- and post-hooks."""
    logger = logging.getLogger(f"{__name__}/{wrap_method.__name__}/{method.__name__})")
    logger.debug("Wrapping method")

    pre_func: Fn[Concatenate[T, ...], None] = (
        EMPTY_FN
        if before is None
        else (before if pass_args else lambda self, *_, **__: before(self))
    )

    post_func: Fn[Concatenate[T, ...], None] = (
        EMPTY_FN
        if after is None
        else (after if pass_args else lambda self, *_, **__: after(self))
    )

    @wraps(method)
    def __wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
        pre_func(self, *args, **kwargs)
        result = method(self, *args, **kwargs)
        post_func(self, *args, **kwargs)
        return result

    return __wrapper


@decorator
def return_namedtuple[**P, T](
    func: Fn[P, tuple[T, ...]],
    /,
    *,
    name: Optional[str] = None,
    field_names: Optional[Sequence[str]] = None,
) -> Fn[P, NTuple[T]]:
    r"""Convert a function's return type to a namedtuple."""
    name = f"{func.__name__}_tuple" if name is None else name
    annotations = getattr(func, "__annotations__", {})
    if "return" not in annotations:
        raise DecoratorError(func, "No return type hint found.")
    return_type = annotations["return"]
    if not issubclass(return_type.__origin__, tuple):
        raise DecoratorError(func, "Return type hint is not a tuple.")

    type_hints = return_type.__args__
    potential_return_names = get_exit_point_names(func)

    if len(type_hints) == 0:
        raise DecoratorError(func, "Return type hint is an empty tuple.")
    if Ellipsis in type_hints:
        raise DecoratorError(func, "Return type hint is a variable length tuple.")
    if field_names is None:
        if len(potential_return_names) != 1:
            raise DecoratorError(func, "Automatic detection of names failed.")
        field_names = potential_return_names.pop()
    elif any(len(r) != len(type_hints) for r in potential_return_names):
        raise DecoratorError(
            func, "Number of names does not match number of return values."
        )

    # create namedtuple
    tuple_type: type[NTuple] = NamedTuple(  # type: ignore[misc]
        name, zip(field_names, type_hints, strict=True)
    )

    @wraps(func)
    def __wrapper(*func_args: P.args, **func_kwargs: P.kwargs) -> NTuple:
        # noinspection PyCallingNonCallable
        return tuple_type(*func(*func_args, **func_kwargs))

    return __wrapper
