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
    "wrap_method",
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
from typing import Any, Concatenate, Optional, cast, overload

from torch import jit, nn

from tsdm.config import CONFIG
from tsdm.utils.types import AnyTypeVar, ClassVar
from tsdm.utils.types import ObjectVar as Obj
from tsdm.utils.types import ParameterVar as P
from tsdm.utils.types import ReturnVar as R
from tsdm.utils.types import TorchModuleVar
from tsdm.utils.types.abc import CollectionType

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


def rpartial(
    func: Callable[P, R], /, *fixed_args: Any, **fixed_kwargs: Any
) -> Callable[..., R]:
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

    def __call__(
        self, *message_lines: str
    ) -> Any:  # FIXME: Return Self DecoratorError:
        r"""Raise a new error."""
        return DecoratorError(self.decorated, message="\n".join(message_lines))

    def __str__(self) -> str:
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
    >>> def wrap_fun(
    ...     func: Callable,
    ...     before: Optional[Callable]=None,
    ...     after: Optional[Callable]=None,
    ...     /,
    ... ) -> Callable:
    ...     '''Wraps function.'''

    Here, there is a problem:

    >>> @wrap_func
    ... def func(x): pass
    ...

    here, and also in the case of wrap_func(func), the result should be an identity operation.
    However, the other case

    >>> @wrap_func(before)
    ... def func(x): pass
    ...
    Traceback (most recent call last):
        ...
    NameError: name 'before' is not defined

    the result is a wrapped function. The fundamental problem is a disambiguation between the cases.
    In either case the decorator sees as input (callable, None, None) and so it cannot distinguish
    whether the first input is a wrapping, or the wrapped.

    Thus, we either need to abandon positional arguments with default values.

    Note however, that it is possible so save the situation by adding at least one
    mandatory positional argument:

    >>> def wrap_fun(
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


def attribute(func: Callable[[Obj], R]) -> R:
    r"""Create decorator that converts method to attribute."""

    @wraps(func, updated=())
    class _attribute:
        __slots__ = ("func", "payload")
        sentinel = object()

        def __init__(self, function: Callable) -> None:
            self.func = function
            self.payload = self.sentinel

        def __get__(self, obj, obj_type=None):
            if obj is None:
                return self
            if self.payload is self.sentinel:
                self.payload = self.func(obj)
            return self.payload

    return cast(R, _attribute(func))


@decorator
def timefun(
    func: Callable[P, R],
    /,
    *,
    append: bool = False,
    loglevel: int = logging.WARNING,
) -> Callable[P, R | tuple[R, float]]:
    r"""Log the execution time of the function. Use as decorator.

    By default, appends the execution time (in seconds) to the function call.

    `outputs, time_elapse = timefun(f, append=True)(inputs)`

    If the function call failed, `outputs=None` and `time_elapsed=float('nan')` are returned.

    If `append=True`, then the decorated function will return a tuple of the form `(func(x), time_elapsed)`.
    """
    timefun_logger = logging.getLogger("timefun")

    @wraps(func)
    def _timed_fun(*args: P.args, **kwargs: P.kwargs) -> R | tuple[R, float]:
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
        except Exception as E:
            timefun_logger.error(
                loglevel, "%s failed with Exception %s", func.__qualname__, E
            )
            RuntimeWarning(f"Function execution failed with Exception {E}")
            raise E
        finally:
            gc.enable()

        return (result, elapsed) if append else result

    return _timed_fun


# @decorator
# def sphinx_value(func: Callable, value: Any, /) -> Callable:
#     r"""Use alternative attribute value during sphinx compilation - useful for attributes."""
#
#     @wraps(func)
#     def _wrapper(*func_args, **func_kwargs):
#         return value
#
#     return _wrapper if os.environ.get("GENERATING_DOCS", False) else func


def trace(func: Callable[P, R]) -> Callable[P, R]:
    r"""Log entering and exiting of function."""
    logger = logging.getLogger("trace")

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
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


def autojit(base_class: type[TorchModuleVar]) -> type[TorchModuleVar]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule:
            ...


        model = jit.script(MyModule())

    and

    .. code-block:: python

        @autojit
        class MyModule:
            ...


        model = MyModule()

    are (roughly?) equivalent
    """
    assert issubclass(base_class, nn.Module)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type,misc]  # pylint: disable=too-few-public-methods
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> TorchModuleVar:  # type: ignore[misc]
            # Note: If __new__() does not return an instance of cls,
            # then the new instance's __init__() method will not be invoked.
            instance: TorchModuleVar = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted: TorchModuleVar = jit.script(instance)
                return scripted
            return instance

    assert issubclass(WrappedClass, base_class)
    return WrappedClass


@decorator
def vectorize(
    func: Callable[[Obj], R],
    /,
    *,
    kind: type[CollectionType],
) -> Callable[[Obj | CollectionType], R | CollectionType]:
    r"""Vectorize a function with a single, positional-only input.

    The signature will change accordingly

    Examples
    --------
    >>> @vectorize(kind=list)
    ... def f(x):
    ...     return x + 1
    ...
    >>> assert f(1) == 2
    >>> assert f(1,2) == [2,3]
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


@overload
def IterItems(obj: ClassVar) -> ClassVar:
    ...


@overload
def IterItems(obj: Obj) -> Obj:
    ...


def IterItems(obj: AnyTypeVar) -> AnyTypeVar:
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
def IterKeys(obj: ClassVar) -> ClassVar:
    ...


@overload
def IterKeys(obj: Obj) -> Obj:
    ...


def IterKeys(obj: AnyTypeVar) -> AnyTypeVar:
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
    func: Callable[P, R],
    /,
    *,
    before: Optional[Callable[[], None] | Callable[P, None]] = None,
    after: Optional[Callable[[], None] | Callable[P, None]] = None,
    pass_args: bool = False,
) -> Callable[P, R]:
    r"""Wrap a function with pre- and post-hooks."""
    if before is None and after is None:
        __logger__.debug("No hooks added to %s", func)
        return func

    if before is not None and after is None:
        __logger__.debug("Adding pre hook %s to %s", before, func)
        if pass_args:

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                before(*args, **kwargs)  # type: ignore[misc]
                return func(*args, **kwargs)

            return _wrapper

        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            before()  # type: ignore[misc]
            return func(*args, **kwargs)

        return _wrapper

    if before is None and after is not None:
        __logger__.debug("Adding post hook %s to %s", after, func)

        if pass_args:

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                result = func(*args, **kwargs)
                after(*args, **kwargs)  # type: ignore[misc]
                return result

            return _wrapper

        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            after()  # type: ignore[misc]
            return result

        return _wrapper

    if before is not None and after is not None:
        __logger__.debug("Adding pre and post hook %s, %s to %s", before, after, func)

        if pass_args:

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                before(*args, **kwargs)  # type: ignore[misc]
                result = func(*args, **kwargs)
                after(*args, **kwargs)  # type: ignore[misc]
                return result

            return _wrapper

        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            before()  # type: ignore[misc]
            result = func(*args, **kwargs)
            after()  # type: ignore[misc]
            return result

        return _wrapper

    raise RuntimeError(f"Unreachable code reached for {func}")


@decorator
def wrap_method(
    func: Callable[Concatenate[Obj, P], R],
    /,
    *,
    before: Optional[
        Callable[[Obj], None] | Callable[Concatenate[Obj, P], None]
    ] = None,
    after: Optional[Callable[[Obj], None] | Callable[Concatenate[Obj, P], None]] = None,
    pass_args: bool = False,
) -> Callable[Concatenate[Obj, P], R]:
    r"""Wrap a function with pre- and post-hooks."""
    if before is None and after is None:
        __logger__.debug("No hooks added to %s", func)
        return func

    if before is not None and after is None:
        __logger__.debug("Adding pre hook %s to %s", before, func)

        if pass_args:

            @wraps(func)
            def _wrapper(self: Obj, *args: P.args, **kwargs: P.kwargs) -> R:
                before(self, *args, **kwargs)  # type: ignore[misc]
                return func(self, *args, **kwargs)

            return _wrapper

        @wraps(func)
        def _wrapper(self: Obj, *args: P.args, **kwargs: P.kwargs) -> R:
            before(self)  # type: ignore[misc]
            return func(self, *args, **kwargs)

        return _wrapper

    if before is None and after is not None:
        __logger__.debug("Adding post hook %s to %s", after, func)

        if pass_args:

            @wraps(func)
            def _wrapper(self: Obj, *args: P.args, **kwargs: P.kwargs) -> R:
                result = func(self, *args, **kwargs)
                after(self, *args, **kwargs)  # type: ignore[misc]
                return result

            return _wrapper

        @wraps(func)
        def _wrapper(self: Obj, *args: P.args, **kwargs: P.kwargs) -> R:
            result = func(self, *args, **kwargs)
            after(self)  # type: ignore[misc]
            return result

        return _wrapper

    if before is not None and after is not None:
        __logger__.debug("Adding pre and post hook %s, %s to %s", before, after, func)

        if pass_args:

            @wraps(func)
            def _wrapper(self: Obj, *args: P.args, **kwargs: P.kwargs) -> R:
                before(self, *args, **kwargs)  # type: ignore[misc]
                result = func(self, *args, **kwargs)
                after(self, *args, **kwargs)  # type: ignore[misc]
                return result

            return _wrapper

        @wraps(func)
        def _wrapper(self: Obj, *args: P.args, **kwargs: P.kwargs) -> R:
            before(self)  # type: ignore[misc]
            result = func(self, *args, **kwargs)
            after(self)  # type: ignore[misc]
            return result

        return _wrapper

    raise RuntimeError(f"Unreachable code reached for {func}")
