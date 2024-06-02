r"""Function decorators for wrapping functions with additional functionality."""

__all__ = [
    # ABCs & Protocols
    # Functions
    "debug",
    "lazy_jit_torch",
    "return_namedtuple",
    "timefun",
    "trace",
    "wrap_func",
    "wrap_method",
]

import gc
import logging
from collections.abc import Callable, Sequence
from functools import wraps
from time import perf_counter_ns

from torch import jit
from typing_extensions import Concatenate, NamedTuple, Optional

from tsdm.types.protocols import NTuple
from tsdm.types.variables import P, R_co, T
from tsdm.utils.decorators.base import DecoratorError, decorator
from tsdm.utils.funcutils import get_exit_point_names

__logger__ = logging.getLogger(__name__)


# region without @decorator ------------------------------------------------------------
def debug(func: Callable[P, R_co], /) -> Callable[P, R_co]:
    r"""Print the function signature and return value."""

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
        args_repr = [f"{type(a)}" for a in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        sign = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({sign})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value

    return __wrapper


def lazy_jit_torch(func: Callable[P, R_co], /) -> Callable[P, R_co]:
    r"""Create decorator to lazily compile a function with `torch.jit.script`."""

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
        # script the original function if it hasn't been scripted yet
        if __wrapper.__scripted is None:  # type: ignore[attr-defined]
            __wrapper.__scripted = jit.script(__wrapper.__original_fn)  # type: ignore[attr-defined]
        return __wrapper.__scripted(*args, **kwargs)  # type: ignore[attr-defined]

    __wrapper.__original_fn = func  # type: ignore[attr-defined]
    __wrapper.__scripted = None  # type: ignore[attr-defined]
    __wrapper.__script_if_tracing_wrapper = True  # type: ignore[attr-defined]
    return __wrapper


def trace(func: Callable[P, R_co], /) -> Callable[P, R_co]:
    r"""Log entering and exiting of function."""
    logger = logging.getLogger("trace")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
        logger.info(
            "%s",
            "\n\t".join((
                f"{func.__qualname__}: ENTERING",
                f"args={tuple(type(arg).__name__ for arg in args)}",
                f"kwargs={str({k: type(v).__name__ for k, v in kwargs.items()})}",
            )),
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
def timefun(
    func: Callable[P, R_co], /, *, append: bool = False, loglevel: int = logging.WARNING
) -> Callable[P, R_co | tuple[R_co, float]]:
    r"""Log the execution time of the function. Use as decorator.

    By default, appends the execution time (in seconds) to the function call.

    `outputs, time_elapse = timefun(f, append=True)(inputs)`

    If the function call failed, `outputs=None` and `time_elapsed=float('nan')` are returned.

    If `append=True`, then the decorated function will return a tuple of the form `(func(x), time_elapsed)`.
    """
    timefun_logger = logging.getLogger("timefun")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co | tuple[R_co, float]:
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

        return (result, elapsed) if append else result

    return __wrapper


@decorator
def wrap_func(
    func: Callable[P, R_co],
    /,
    *,
    before: Optional[Callable[..., None]] = None,
    after: Optional[Callable[..., None]] = None,
    pass_args: bool = False,
) -> Callable[P, R_co]:
    r"""Wrap a function with pre- and post-hooks."""
    logger = __logger__.getChild(func.__name__)

    match before, after, pass_args:
        case None, None, bool():
            logger.debug("No hooks to add, returning as-is.")
            return func

        case Callable() as pre, None, True:  # type: ignore[misc]
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
                pre(*args, **kwargs)
                return func(*args, **kwargs)

        case None, Callable() as post, True:  # type: ignore[misc]
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
                result = func(*args, **kwargs)
                post(*args, **kwargs)
                return result

        case Callable() as pre, Callable() as post, True:  # type: ignore[misc]
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
                pre(*args, **kwargs)
                result = func(*args, **kwargs)
                post(*args, **kwargs)
                return result

        case Callable() as pre, None, False:  # type: ignore[misc]
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
                pre()
                return func(*args, **kwargs)

        case None, Callable() as post, False:  # type: ignore[misc]
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
                result = func(*args, **kwargs)
                post()
                return result

        case Callable() as pre, Callable() as post, False:  # type: ignore[misc]
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
                pre()
                result = func(*args, **kwargs)
                post()
                return result

        case _:
            raise TypeError("Got unexpected arguments.")

    return __wrapper  # type: ignore[unreachable]


@decorator
def wrap_method(
    func: Callable[Concatenate[T, P], R_co],
    /,
    *,
    before: Optional[Callable[Concatenate[T, ...], None]] = None,
    after: Optional[Callable[Concatenate[T, ...], None]] = None,
    pass_args: bool = False,
) -> Callable[Concatenate[T, P], R_co]:
    r"""Wrap a function with pre- and post-hooks."""
    logger = __logger__.getChild(func.__name__)

    match before, after, pass_args:
        case None, None, bool():
            logger.debug("No hooks to add, returning as-is.")
            return func

        case Callable() as pre, None, True:  # type: ignore[misc]
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            # Q: Why do we have to add pyright ignore in this one, but not the others?
            @wraps(func)
            def __wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R_co:  # pyright: ignore[reportRedeclaration]
                pre(self, *args, **kwargs)
                return func(self, *args, **kwargs)

        case None, Callable() as post, True:  # type: ignore[misc]
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
                result = func(self, *args, **kwargs)
                post(self, *args, **kwargs)
                return result

        case Callable() as pre, Callable() as post, True:  # type: ignore[misc]
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
                pre(self, *args, **kwargs)
                result = func(self, *args, **kwargs)
                post(self, *args, **kwargs)
                return result

        case Callable() as pre, None, False:  # type: ignore[misc]
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
                pre(self)
                return func(self, *args, **kwargs)

        case None, Callable() as post, False:  # type: ignore[misc]
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
                result = func(self, *args, **kwargs)
                post(self)
                return result

        case Callable() as pre, Callable() as post, False:  # type: ignore[misc]
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def __wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R_co:
                pre(self)
                result = func(self, *args, **kwargs)
                post(self)
                return result

        case _:
            raise TypeError("Got unexpected arguments.")

    return __wrapper  # type: ignore[unreachable]


@decorator
def return_namedtuple(
    func: Callable[P, tuple[T, ...]],
    /,
    *,
    name: Optional[str] = None,
    field_names: Optional[Sequence[str]] = None,
) -> Callable[P, NTuple[T]]:
    r"""Convert a function's return type to a namedtuple."""
    name = f"{func.__name__}_tuple" if name is None else name
    annotations = getattr(func, "__annotations__", {})
    return_type = annotations.get("return", NotImplemented)
    if return_type is NotImplemented:
        raise DecoratorError(func, "No return type hint found.")
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
