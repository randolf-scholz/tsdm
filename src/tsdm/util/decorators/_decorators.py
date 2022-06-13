r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

__all__ = [
    # Classes
    # Functions
    "decorator",
    # "sphinx_value",
    "timefun",
    "trace",
    "vectorize",
    "wrap_func",
    # Class Decorators
    "autojit",
    "IterItems",
    "IterKeys",
    # Exceptions
    "DecoratorError",
]

import gc
import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, Signature, signature
from time import perf_counter_ns
from typing import Any, Optional, Union, overload

from torch import jit, nn

from tsdm.config import conf
from tsdm.util.types import (
    ClassType,
    ObjectType,
    Parameters,
    ReturnType,
    Type,
    nnModuleType,
)
from tsdm.util.types.abc import CollectionType

__logger__ = logging.getLogger(__name__)

KEYWORD_ONLY = Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = Parameter.VAR_KEYWORD
VAR_POSITIONAL = Parameter.VAR_POSITIONAL
EMPTY = Parameter.empty
_DECORATED = object()

PARAM_TYPES = (
    (POSITIONAL_ONLY, True),
    (POSITIONAL_ONLY, False),
    (POSITIONAL_OR_KEYWORD, True),
    (POSITIONAL_OR_KEYWORD, False),
    (VAR_POSITIONAL, True),
    (KEYWORD_ONLY, True),
    (KEYWORD_ONLY, False),
    (VAR_KEYWORD, True),
)


def rpartial(func: Callable, /, *fixed_args: Any, **fixed_kwargs: Any) -> Callable:
    r"""Apply positional arguments from the right."""

    @wraps(func)
    def _wrapper(*func_args, **func_kwargs):
        return func(*(func_args + fixed_args), **(func_kwargs | fixed_kwargs))

    return _wrapper


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
        max_key_len = max(9, max(len(key) for key in sign.parameters))
        max_kind_len = max(len(str(param.kind)) for param in sign.parameters.values())
        default_message: tuple[str, ...] = (
            f"Signature: {sign}",
            "\n".join(
                f"{key.ljust(max_key_len)}: {str(param.kind).ljust(max_kind_len)}"
                f", Optional: {param.default is EMPTY}"
                for key, param in sign.parameters.items()
            ),
            self.message,
        )
        return super().__str__() + "\n" + "\n".join(default_message)


def _last_positional_only_arg_index(sig: Signature) -> int:
    r"""Return index such that all parameters before are POSITIONAL_ONLY."""
    for i, param in enumerate(sig.parameters.values()):
        if param.kind is not POSITIONAL_ONLY:
            return i
    return len(sig.parameters)


def decorator(deco: Callable) -> Callable:
    r"""Meta-Decorator for constructing parametrized decorators.

    There are 3 different ways of using decorators:

    1. BARE MODE
        >>> @deco
        ... def func(*args, **kwargs):
        ...     # Input: func
        ...     # Output: Wrapped Function
    2. FUNCTIONAL MODE
        >>> deco(func, *args, **kwargs)
        ...     # Input: func, args, kwargs
        ...     # Output: Wrapped Function
    3. BRACKET MODE
        >>> @deco(*args, **kwargs)
        ... def func(*args, **kwargs):
        ...     # Input: args, kwargs
        ...     # Output: decorator with single positional argument

    Crucially, one needs to be able to distinguish between the three modes.
    In particular, when the decorator has optional arguments, one needs to be able to distinguish
    between FUNCTIONAL MODE and BRACKET MODE.

    To achieve this, we introduce a special senitel value for the first argument.
    Adding this senitel requires that the decorator has no mandatory positional-only arguments.
    Otherwise, the new signature would have an optional positional-only argument before the first
    mandatory positional-only argument.

    Therefore, we add senitel values to all mandatory positional-only arguments.
    If the mandatory positional args are not given

    IDEA: We replace the decorators signature with a new signature in which all arguments
    have default values.

    Fundamentally, signatures that lead to ambiguity between the 3 modes cannot be allowed.
    Note that in BARE MODE, the decorator receives no arguments.
    In BRACKET MODE, the decorator receives the arguments as given, and must return a
    decorator that takes a single input.

    +------------+-----------------+----------+-------------------+
    |            | mandatory args? | VAR_ARGS | no mandatory args |
    +============+=================+==========+===================+
    | bare       | ✘               | ✘        | ✔                 |
    +------------+-----------------+----------+-------------------+
    | bracket    | ✔               | ✔        | ✔                 |
    +------------+-----------------+----------+-------------------+
    | functional | ✔               | ✔        | ✔                 |
    +------------+-----------------+----------+-------------------+

    Examples
    --------
    >>> def wrap_func(
    ...     func: Callable,
    ...     before: Optional[Callable]=None,
    ...     after: Optional[Callable]=None,
    ...     /
    ... ) -> Callable:
    ...     '''Wraps function.'''

    Here, there is a problem:

    >>> @wrap_func
    ... def func(...)

    here, and also in the case of wrap_func(func), the result should be an identity operation.
    However, the other case

    >>> @wrap_func(before)
    ...def func(...)

    the result is a wrapped function. The fundamental problem is a disambiguation between the cases.
    In either case the decorator sees as input (callable, None, None) and so it cannot distinguish
    whether the first input is a wrapping, or the wrapped.

    Thus we either need to abandon positional arguments with default values.

    Note however, that it is possible so save the situation by adding at least one
    mandatory positional argument:

    >>> def wrap_func(
    ...     func: Callable,
    ...     before: Optional[Callable],
    ...     after: Optional[Callable]=None,
    ...     /
    ... ) -> Callable:
    ...     '''Wraps function.'''

    Because now, we can test how many arguments were passed. If only a single positional argument was passed,
    we know that the decorator was called in the bracket mode.

    Arguments::

        PO | PO+D | *ARGS | PO/K | PO/K+D | KO | KO+D | **KWARGS |

    For decorators, we only allow signatures of the form::

        func | PO | KO | KO + D | **KWARGS

    Under the hood, the signature will be changed to::

        PO | __func__ = None | / | * | KO | KO + D | **KWARGS

    I.e. we insert a single positional only argument with default, which is the function to be wrapped.
    """
    __logger__.debug(">>>>> Creating decorator %s <<<<<", deco)

    deco_sig = signature(deco)
    ErrorHandler = DecoratorError(deco)
    # param_iterator = iter(deco_sig.parameters.items())

    BUCKETS: dict[Any, set[str]] = {key: set() for key in PARAM_TYPES}

    for key, param in deco_sig.parameters.items():
        BUCKETS[param.kind, param.default is EMPTY].add(key)

    __logger__.debug(
        "DETECTED SIGNATURE:"
        "\n\t%s POSITIONAL_ONLY       (mandatory)"
        "\n\t%s POSITIONAL_ONLY       (optional)"
        "\n\t%s POSITIONAL_OR_KEYWORD (mandatory)"
        "\n\t%s POSITIONAL_OR_KEYWORD (optional)"
        "\n\t%s VAR_POSITIONAL"
        "\n\t%s KEYWORD_ONLY          (mandatory)"
        "\n\t%s KEYWORD_ONLY          (optional)"
        "\n\t%s VAR_KEYWORD",
        *(len(BUCKETS[key]) for key in PARAM_TYPES),
    )

    if BUCKETS[POSITIONAL_OR_KEYWORD, True] or BUCKETS[POSITIONAL_OR_KEYWORD, False]:
        raise ErrorHandler(
            "Decorator does not support POSITIONAL_OR_KEYWORD arguments!!",
            "Separate positional and keyword arguments using '/' and '*':"
            ">>> def deco(func, /, *, ko1, ko2, **kwargs): ...",
            "Cf. https://www.python.org/dev/peps/pep-0570/",
        )
    if BUCKETS[POSITIONAL_ONLY, False]:
        raise ErrorHandler(
            "Decorator does not support POSITIONAL_ONLY arguments with defaults!!"
        )
    if not len(BUCKETS[POSITIONAL_ONLY, True]) == 1:
        raise ErrorHandler(
            "Decorator must have exactly 1 POSITIONAL_ONLY argument: the function to be decorated."
            ">>> def deco(func, /, *, ko1, ko2, **kwargs): ...",
        )
    if BUCKETS[VAR_POSITIONAL, True]:
        raise ErrorHandler("Decorator does not support VAR_POSITIONAL arguments!!")

    # (1b) modify the signature to add a new parameter '__func__' as the single
    # positional-only argument with a default value.
    # params = list(deco_sig.parameters.values())
    # index = _last_positional_only_arg_index(deco_sig)
    # params.insert(
    #     index, Parameter("__func__", kind=Parameter.POSITIONAL_ONLY, default=_DECORATED)
    # )
    # del params[0]
    # new_sig = deco_sig.replace(parameters=params)

    @wraps(deco)
    def _parametrized_decorator(
        __func__: Optional[Callable] = None, *args: Any, **kwargs: Any
    ) -> Callable:
        __logger__.debug(
            "DECORATING \n\tfunc=%s: \n\targs=%s, \n\tkwargs=%s", deco, args, kwargs
        )

        if __func__ is None:
            __logger__.debug("%s: Decorator used in BRACKET mode.", deco)
            return rpartial(deco, *args, **kwargs)

        assert callable(__func__), "First argument must be callable!"
        __logger__.debug("%s: Decorator in FUNCTIONAL/BARE mode.", deco)
        return deco(*(__func__, *args), **kwargs)

    return _parametrized_decorator


@decorator
def timefun(
    fun: Callable[Parameters, ReturnType],
    /,
    *,
    append: bool = False,
    loglevel: int = logging.WARNING,
) -> Callable[Parameters, ReturnType]:
    r"""Log the execution time of the function. Use as decorator.

    By default appends the execution time (in seconds) to the function call.

    `outputs, time_elapse = timefun(f, append=True)(inputs)`

    If the function call failed, `outputs=None` and `time_elapsed=float('nan')` are returned.

    Parameters
    ----------
    fun: Callable
    append: bool, default True
        Whether to append the time result to the function call
    loglevel: int, default logging.Warning (20)
    """
    timefun_logger = logging.getLogger("timefun")

    @wraps(fun)
    def _timed_fun(*args, **kwargs):
        gc.collect()
        gc.disable()
        try:
            start_time = perf_counter_ns()
            result = fun(*args, **kwargs)
            end_time = perf_counter_ns()
            elapsed = (end_time - start_time) / 10**9
            timefun_logger.log(
                loglevel, "%s executed in %.4f s", fun.__qualname__, elapsed
            )
        except (KeyboardInterrupt, SystemExit) as E:
            raise E
        except Exception as E:  # pylint: disable=W0703
            result = None
            elapsed = float("nan")
            RuntimeWarning(f"Function execution failed with Exception {E}")
            timefun_logger.log(
                loglevel, "%s failed with Exception %s", fun.__qualname__, E
            )
        gc.enable()

        return (result, elapsed) if append else result

    return _timed_fun


# @decorator
# def sphinx_value(func: Callable, value: Any, /) -> Callable:
#     """Use alternative attribute value during sphinx compilation - useful for attributes.
#
#     Parameters
#     ----------
#     func: Callable
#     value: Any
#
#     Returns
#     -------
#     Callable
#     """
#
#     @wraps(func)
#     def _wrapper(*func_args, **func_kwargs):
#         return value
#
#     return _wrapper if os.environ.get("GENERATING_DOCS", False) else func


def trace(func: Callable[Parameters, ReturnType]) -> Callable[Parameters, ReturnType]:
    r"""Log entering and exiting of function.

    Parameters
    ----------
    func: Callable

    Returns
    -------
    Callable
    """
    logger = logging.getLogger("trace")

    @wraps(func)
    def _wrapper(*args, **kwargs):
        logger.info(
            "%s",
            "\n\t".join(
                (
                    f"{func.__qualname__}: ENTERING",
                    f"args={tuple(type(arg).__name__ for arg in args)}",
                    f"kwargs={str({k:type(v).__name__ for k,v in kwargs.items()})}",
                )
            ),
        )
        try:
            logger.info("%s: EXECUTING", func.__qualname__)
            result = func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit) as E:
            raise E
        except Exception as E:
            logger.error("%s: FAILURE with Exception %s", func.__qualname__, E)
            raise RuntimeError(f"Function execution failed with Exception {E}") from E
        else:
            logger.info(
                "%s: SUCCESS with result=%s", func.__qualname__, type(result).__name__
            )
        logger.info("%s", "\n\t".join((f"{func.__qualname__}: EXITING",)))
        return result

    return _wrapper


def autojit(base_class: type[nnModuleType]) -> type[nnModuleType]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule():
            ...

        model = jit.script(MyModule())

    and

    .. code-block:: python

        @autojit
        class MyModule():
            ...

        model = MyModule()

    are (roughly?) equivalent

    Parameters
    ----------
    base_class: type[nn.Module]

    Returns
    -------
    type
    """
    assert issubclass(base_class, nn.Module)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore  # pylint: disable=too-few-public-methods
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> nnModuleType:  # type: ignore[misc]
            # Note: If __new__() does not return an instance of cls,
            # then the new instance's __init__() method will not be invoked.
            instance: nnModuleType = base_class(*args, **kwargs)

            if conf.autojit:
                scripted: nnModuleType = jit.script(instance)
                return scripted
            return instance

    assert issubclass(WrappedClass, base_class)
    return WrappedClass


@decorator
def vectorize(
    func: Callable[[ObjectType], ReturnType],
    /,
    *,
    kind: type[CollectionType],
) -> Callable[[Union[ObjectType, CollectionType]], Union[ReturnType, CollectionType]]:
    r"""Vectorize a function with a single, positional-only input.

    The signature will change accordingly

    Parameters
    ----------
    func: Callable[[ObjectType], ReturnType]
    kind: type[CollectionType]

    Returns
    -------
    Callable[[Union[ObjectType, CollectionType]], Union[ReturnType, CollectionType]]

    Examples
    --------
    .. code-block:: python

        @vectorize(list)
        def f(x):
            return x + 1

        assert f(1) == 2
        assert f(1,2) == [2,3]
    """
    params = list(signature(func).parameters.values())

    if not params:
        raise ValueError(f"{func} has no parameters")
    if params[0].kind not in (
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(f"{func} must have a single positional parameter!")
    for param in params[1:]:
        if param.kind not in (Parameter.KEYWORD_ONLY, Parameter.VAR_KEYWORD):
            raise ValueError(f"{func} must have a single positional parameter!")

    @wraps(func)
    def _wrapper(arg, *args):
        if not args:
            return func(arg)
        return kind(func(x) for x in (arg, *args))  # type: ignore[call-arg]

    return _wrapper


# @decorator
# def apply_on_iter(
#     cls: Iterable[ObjectType], func: Callable[[ObjectType], ReturnType]
# ) -> Iterable[ReturnType]:
#
#     iter_method = getattr(cls, "__iter__")
#
#     @wraps(iter_method)
#     def wrapper() -> Iterable[ReturnType]:
#         for x in iter_method():
#             yield func(x)
#
#     setattr(cls, "__iter__", wrapper)
#
#     return cls


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


@overload
def IterItems(obj: ClassType) -> ClassType:
    ...


@overload
def IterItems(obj: ObjectType) -> ObjectType:
    ...


def IterItems(obj: Type) -> Type:
    r"""Wrap a class such that `__getitem__` returns (key, value) pairs."""
    base_class = obj if isinstance(obj, type) else type(obj)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type:ignore[valid-type, misc]
        r"""A simple Wrapper."""

        def __getitem__(self, key: Any) -> tuple[Any, Any]:
            r"""Get the item from the dataset."""
            return key, super().__getitem__(key)

        def __repr__(self) -> str:
            r"""Representation of the dataset."""
            return r"IterItems@" + super().__repr__()

    if isinstance(obj, type):
        return WrappedClass  # type: ignore[return-value]
    obj = deepcopy(obj)
    obj.__class__ = WrappedClass
    return obj


@overload
def IterKeys(obj: ClassType) -> ClassType:
    ...


@overload
def IterKeys(obj: ObjectType) -> ObjectType:
    ...


def IterKeys(obj: Type) -> Type:
    r"""Wrap a class such that `__getitem__` returns key instead."""
    base_class = obj if isinstance(obj, type) else type(obj)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type:ignore[valid-type, misc]
        r"""A simple Wrapper."""

        def __getitem__(self, key: Any) -> tuple[Any, Any]:
            r"""Return the key as is."""
            return key

        def __repr__(self) -> str:
            r"""Representation of the new object."""
            return r"IterKeys@" + super().__repr__()

    if isinstance(obj, type):
        return WrappedClass  # type: ignore[return-value]
    obj = deepcopy(obj)
    obj.__class__ = WrappedClass
    return obj


@decorator
def wrap_func(
    func: Callable[Parameters, ReturnType],
    /,
    *,
    before: Optional[Callable[Parameters, Any]] = None,
    after: Optional[Callable[Parameters, Any]] = None,
) -> Callable[Parameters, ReturnType]:
    r"""Wrap a function with pre and post hooks."""
    if before is None and after is None:
        __logger__.debug("No hooks added to %s", func)
        return func

    if before is not None and after is None:
        __logger__.debug("Adding pre hook %s to %s", before, func)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            before(*args, **kwargs)  # type: ignore[misc]
            result = func(*args, **kwargs)
            return result

        return _wrapper

    if before is None and after is not None:
        __logger__.debug("Adding post hook %s to %s", after, func)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            after(*args, **kwargs)  # type: ignore[misc]
            return result

        return _wrapper

    if before is not None and after is not None:
        __logger__.debug("Adding pre hook %s to %s", before, func)
        __logger__.debug("Adding post hook %s to %s", after, func)

        @wraps(func)
        def _wrapper(*args, **kwargs):
            before(*args, **kwargs)  # type: ignore[misc]
            result = func(*args, **kwargs)
            after(*args, **kwargs)  # type: ignore[misc]
            return result

        return _wrapper

    raise RuntimeError(f"Unreachable code reached for {func}")


# @decorator
# def pre_hook(func: Callable, hook: Callable, /) -> Callable:
#     r"""Wrap a function with a pre wrap_hook."""
#
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         r"""Wrap a function with a pre wrap_hook."""
#         hook(*args, **kwargs)
#         return func(*args, **kwargs)
#
#     return wrapper
#
#
# @decorator
# def post_hook(func: Callable, hook: Callable, /) -> Callable:
#     r"""Wrap a function with a post wrap_hook."""
#
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         r"""Wrap a function with a post wrap_hook."""
#         result = func(*args, **kwargs)
#         hook(*args, **kwargs)
#         return result
#
#     return wrapper
#
#
# @decorator
# def wrap_chain(
#     func: Callable,
#     /,
#     *,
#     before: Optional[Callable] = None,
#     after: Optional[Callable] = None,
# ) -> Callable:
#     """Chain a function with pre and post func.
#
#     Parameters
#     ----------
#     func
#     before
#     after
#
#     Returns
#     -------
#     Callable
#     """
#
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         r"""Wrap a function with pre and post hooks."""
#         pre_result = before(*args, **kwargs)
#         result = func(pre_result)
#         post_result = after(result)
#         return post_result
#
#     return wrapper


# @overload
# def wrapmethod(obj: ClassType, method: str, func: Callable) -> ClassType:
#     ...
#
#
# @overload
# def wrapmethod(obj: ObjectType, method: str, func: Callable) -> ObjectType:
#     ...
#
#
# @decorator
# def wrapmethod(obj, method, func, /):
#     r"""Wrap a method of a class/instance or instance."""
#     if isinstance(obj, type):
#         base_class = obj
#     else:
#         base_class = type(obj)
#
#     @wraps(base_class, updated=())
#     class WrappedClass(base_class):
#         r"""A simple Wrapper."""
#
#         def __repr__(self) -> str:
#             r"""Representation of the dataset."""
#             return f"wrapmethod[{method}, {func.__name__}]@" + super().__repr__()
#
#     setattr(WrappedClass, method, MethodType(func, obj))
#
#     if isinstance(obj, type):
#         return WrappedClass
#
#     obj = deepcopy(obj)  # <--- do we need this?
#     obj.__class__ = WrappedClass
#     return obj
