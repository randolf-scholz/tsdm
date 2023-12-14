r"""Submodule containing general purpose decorators."""

__all__ = [
    # Classes
    "Decorator",
    "ClassDecorator",
    # Functions
    "decorator",
    "debug",
    # "sphinx_value",
    "lazy_torch_jit",
    "return_namedtuple",
    "timefun",
    "trace",
    "vectorize",
    "wrap_func",
    "wrap_method",
    # Class Decorators
    # Exceptions
    "DecoratorError",
]

import ast
import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, Signature, getsource, signature
from time import perf_counter_ns
from types import GenericAlias

from torch import jit
from typing_extensions import (
    Any,
    Concatenate,
    NamedTuple,
    Optional,
    ParamSpec,
    Protocol,
    Self,
    cast,
)

from tsdm.types.aliases import Nested
from tsdm.types.callback_protocols import Func
from tsdm.types.protocols import NTuple
from tsdm.types.variables import CollectionType, any_var as T, return_var_co as R
from tsdm.utils.funcutils import rpartial

__logger__: logging.Logger = logging.getLogger(__name__)

P = ParamSpec("P")

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


class Decorator(Protocol):
    r"""Decorator Protocol."""

    def __call__(self, func: Callable, /) -> Callable:
        r"""Decorate a function."""
        ...


class ClassDecorator(Protocol):
    r"""Class Decorator Protocol."""

    def __call__(self, cls: type, /) -> type:
        r"""Decorate a class."""
        ...


def collect_exit_points(func: Callable) -> list[ast.Return]:
    """Collect all exit points of a function as ast nodes."""
    tree = ast.parse(getsource(func))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Return)]


def exit_point_names(func: Callable) -> list[tuple[str, ...]]:
    """Return the variable names used in exit nodes."""
    exit_points = collect_exit_points(func)

    var_names = []
    for exit_point in exit_points:
        assert isinstance(exit_point.value, ast.Tuple)

        e: tuple[str, ...] = ()
        for obj in exit_point.value.elts:
            assert isinstance(obj, ast.Name)
            e += (obj.id,)
        var_names.append(e)
    return var_names


@dataclass
class DecoratorError(Exception):
    r"""Raise Error related to decorator construction."""

    decorated: Callable
    r"""The decorator."""
    message: str = ""
    r"""Default message to print."""

    def __call__(self: Self, *message_lines: str) -> Self:
        r"""Raise a new error."""
        # TODO: CHECK if dataclasses are the problem
        return DecoratorError(self.decorated, message="\n".join(message_lines))  # type: ignore[return-value]

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
    r"""Returns last index for which all preceeding parameters are POSITIONAL_ONLY."""
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

    To achieve this, we introduce a special sentinel value for the first argument.
    Adding this senitel requires that the decorator has no mandatory positional-only arguments.
    Otherwise, the new signature would have an optional positional-only argument before the first
    mandatory positional-only argument.

    Therefore, we add sentinel values to all mandatory positional-only arguments.
    If the mandatory positional args are not given

    IDEA: We replace the decorator's signature with a new signature in which all arguments
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
    In either case, the decorator sees as input (callable, None, None) and so it cannot distinguish
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

    error_msg = (
        "DETECTED SIGNATURE:"  # pylint: disable=consider-using-f-string
        "\n\t %s: %s     (mandatory)"
        "\n\t %s: %s     (optional)"
        "\n\t %s: %s     (mandatory)"
        "\n\t %s: %s     (optional)"
        "\n\t %s: %s     (optional)"
        "\n\t %s: %s     (mandatory)"
        "\n\t %s: %s     (optional)"
        "\n\t %s: %s     (optional)"
    ).format(*(x for pair in [(k, len(v)) for k, v in BUCKETS.items()] for x in pair))

    if BUCKETS[POSITIONAL_OR_KEYWORD, True] or BUCKETS[POSITIONAL_OR_KEYWORD, False]:
        raise ErrorHandler(
            "Decorator does not support POSITIONAL_OR_KEYWORD arguments!!",
            "Separate positional and keyword arguments using '/' and '*':"
            ">>> def deco(func, /, *, ko1, ko2, **kwargs): ...",
            "Cf. https://www.python.org/dev/peps/pep-0570/",
            error_msg,
        )
    if BUCKETS[POSITIONAL_ONLY, False]:
        raise ErrorHandler(
            "Decorator does not support POSITIONAL_ONLY arguments with defaults!!",
            error_msg,
        )
    if not len(BUCKETS[POSITIONAL_ONLY, True]) == 1:
        raise ErrorHandler(
            "Decorator must have exactly 1 POSITIONAL_ONLY argument: the function to be"
            " decorated.>>> def deco(func, /, *, ko1, ko2, **kwargs): ...",
            error_msg,
        )
    if BUCKETS[VAR_POSITIONAL, True]:
        raise ErrorHandler(
            "Decorator does not support VAR_POSITIONAL arguments!!",
            error_msg,
        )

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


def attribute(func: Callable[[T], R]) -> R:
    r"""Create a decorator that converts method to attribute."""

    @wraps(func, updated=())
    class _attribute:
        __slots__ = ("func", "payload")
        sentinel = object()
        func: Callable[[T], R]
        payload: R

        def __init__(self, function: Callable) -> None:
            self.func = function
            self.payload = cast(Any, self.sentinel)

        def __get__(self, obj: T | None, obj_type: Optional[type] = None) -> Self | R:
            if obj is None:
                return self
            if self.payload is self.sentinel:
                self.payload = self.func(obj)
            return self.payload

    return cast(R, _attribute(func))


def debug(func: Callable[P, R]) -> Callable[P, R]:
    """Print the function signature and return value."""

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args_repr = [f"{type(a)}" for a in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        sign = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({sign})")
        value = func(*args, **kwargs)
        # print(f"{func.__name__!r} returned {value!r}")
        return value

    return _wrapper


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
        except Exception as exc:
            timefun_logger.error(
                loglevel, "%s failed with Exception %s", func.__qualname__, exc
            )
            raise RuntimeError("Function execution failed") from exc
        finally:
            gc.enable()

        return (result, elapsed) if append else result

    return _timed_fun


def trace(func: Callable[P, R]) -> Callable[P, R]:
    r"""Log entering and exiting of function."""
    logger = logging.getLogger("trace")

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.info(
            "%s",
            "\n\t".join((
                f"{func.__qualname__}: ENTERING",
                f"args={tuple(type(arg).__name__ for arg in args)}",
                f"kwargs={str({k:type(v).__name__ for k,v in kwargs.items()})}",
            )),
        )
        try:
            logger.info("%s: EXECUTING", func.__qualname__)
            result = func(*args, **kwargs)
        except Exception as exc:
            logger.error("%s: FAILURE with Exception %s", func.__qualname__, exc)
            raise RuntimeError(
                f"Function execution failed with Exception {exc}"
            ) from exc
        logger.info(
            "%s: SUCCESS with result=%s", func.__qualname__, type(result).__name__
        )
        logger.info("%s", "\n\t".join((f"{func.__qualname__}: EXITING",)))
        return result

    return _wrapper


@decorator
def vectorize(
    func: Callable[[T], R],
    /,
    *,
    kind: type[CollectionType],
) -> Callable[[T | CollectionType], R | CollectionType]:
    r"""Vectorize a function with a single, positional-only input.

    The signature will change accordingly.

    Examples:
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
    logger = __logger__.getChild(func.__name__)

    # FIXME: https://github.com/python/mypy/issues/6680
    match before, after, pass_args:
        case None, None, bool():
            logger.debug("No hooks to add, returning as-is.")
            return func

        case Func() as pre, None, True:
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                pre(*args, **kwargs)
                return func(*args, **kwargs)

        case None, Func() as post, True:
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                result = func(*args, **kwargs)
                post(*args, **kwargs)
                return result

        case Func() as pre, Func() as post, True:
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                pre(*args, **kwargs)
                result = func(*args, **kwargs)
                post(*args, **kwargs)
                return result

        case Func() as pre, None, False:
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                pre()
                return func(*args, **kwargs)

        case None, Func() as post, False:
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                result = func(*args, **kwargs)
                post()
                return result

        case Func() as pre, Func() as post, False:
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                pre()
                result = func(*args, **kwargs)
                post()
                return result

        case _:
            raise TypeError("Got unexpected arguments.")

    return _wrapper  # type: ignore[unreachable]


@decorator
def wrap_method(
    func: Callable[Concatenate[T, P], R],
    /,
    *,
    before: Optional[Callable[[T], None] | Callable[Concatenate[T, P], None]] = None,
    after: Optional[Callable[[T], None] | Callable[Concatenate[T, P], None]] = None,
    pass_args: bool = False,
) -> Callable[Concatenate[T, P], R]:
    r"""Wrap a function with pre- and post-hooks."""
    logger = __logger__.getChild(func.__name__)

    match before, after, pass_args:
        case None, None, bool():
            logger.debug("No hooks to add, returning as-is.")
            return func

        case Func() as pre, None, True:
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            @wraps(func)
            def _method(  # pyright: ignore
                self: T, *args: P.args, **kwargs: P.kwargs
            ) -> R:
                pre(self, *args, **kwargs)
                return func(self, *args, **kwargs)

        case None, Func() as post, True:
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def _method(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
                result = func(self, *args, **kwargs)
                post(self, *args, **kwargs)
                return result

        case Func() as pre, Func() as post, True:
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def _method(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
                pre(self, *args, **kwargs)
                result = func(self, *args, **kwargs)
                post(self, *args, **kwargs)
                return result

        case Func() as pre, None, False:
            logger.debug("Adding pre hook %s", pre)  # type: ignore[unreachable]

            @wraps(func)
            def _method(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
                pre(self)
                return func(self, *args, **kwargs)

        case None, Func() as post, False:
            logger.debug("Adding post hook %s", post)  # type: ignore[unreachable]

            @wraps(func)
            def _method(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
                result = func(self, *args, **kwargs)
                post(self)
                return result

        case Func() as pre, Func() as post, False:
            logger.debug("Adding pre hook %s and post hook %s", pre, post)  # type: ignore[unreachable]

            @wraps(func)
            def _method(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
                pre(self)
                result = func(self, *args, **kwargs)
                post(self)
                return result

        case _:
            raise TypeError("Got unexpected arguments.")

    return _method  # type: ignore[unreachable]


def lazy_torch_jit(func: Callable[P, R]) -> Callable[P, R]:
    """Create decorator to lazily compile a function with `torch.jit.script`."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # script the original function if it hasn't been scripted yet
        if wrapper.__scripted is None:  # type: ignore[attr-defined]
            wrapper.__scripted = jit.script(wrapper.__original_fn)  # type: ignore[attr-defined]
        return wrapper.__scripted(*args, **kwargs)  # type: ignore[attr-defined]

    wrapper.__original_fn = func  # type: ignore[attr-defined]
    wrapper.__scripted = None  # type: ignore[attr-defined]
    wrapper.__script_if_tracing_wrapper = True  # type: ignore[attr-defined]
    return wrapper


@decorator
def return_namedtuple(
    func: Callable[P, tuple],
    /,
    *,
    name: Optional[str] = None,
    field_names: Optional[Sequence[str]] = None,
) -> Callable[P, NTuple]:
    """Convert a function's return type to a namedtuple."""
    name = f"{func.__name__}_tuple" if name is None else name

    # noinspection PyUnresolvedReferences
    return_type: GenericAlias = func.__annotations__.get("return", NotImplemented)
    if return_type is NotImplemented:
        raise DecoratorError(func, "No return type hint found.")
    if not issubclass(return_type.__origin__, tuple):
        raise DecoratorError(func, "Return type hint is not a tuple.")

    type_hints = return_type.__args__
    potential_return_names = set(exit_point_names(func))

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
    tuple_type: type[NTuple] = NamedTuple(name, zip(field_names, type_hints))  # type: ignore[misc]

    @wraps(func)
    def _wrapper(*func_args: P.args, **func_kwargs: P.kwargs) -> NTuple:
        # noinspection PyCallingNonCallable
        return tuple_type(*func(*func_args, **func_kwargs))

    return _wrapper


def recurse_on_builtin_container(
    func: Callable[[T], R],
    /,
    *,
    kind: type[T],
) -> Callable[[Nested[T]], Nested[R]]:
    r"""Apply function to nested iterables of a given kind.

    Args:
        kind: The type of the leave nodes
        func: A function to apply to all leave Nodes
    """
    if issubclass(kind, (tuple, list, set, frozenset, dict)):
        raise TypeError(f"kind must not be a builtin container! Got {kind=}")

    @wraps(func)
    def recurse(x: Nested[T]) -> Nested[R]:
        match x:
            case kind():  # type: ignore[misc]
                return func(x)  # type: ignore[unreachable]
            case dict() as Dict:
                return {k: recurse(v) for k, v in Dict.items()}
            case list() as List:
                return [recurse(obj) for obj in List]
            case tuple() as Tuple:
                return tuple(recurse(obj) for obj in Tuple)
            case set() as Set:
                return {recurse(obj) for obj in Set}  # pyright: ignore
            case frozenset() as FrozenSet:
                return frozenset(recurse(obj) for obj in FrozenSet)
            case _:
                raise TypeError(f"Unsupported type: {type(x)}")

    return recurse


# def extends(parent_func: Callable[P, None], /) -> Callable[[Callable], Callable]:
#     """Decorator to extend a parent function.
#
#     For example, when one wants to extend the __init__ of a parent class
#     with an extra argument.
#
#     This will synthesize a new function that combines the extra arguments with
#     the ones of the parent function. The new arguments passed to the synthesized
#     function are available within the function body.
#
#     Example:
#
#     class Parent:
#         def foo(self, a, b, /, *, key):
#             ...
#
#     class Child(Parent):
#
#         @extends(Parent.foo)
#         def foo(self, c, *parent_args, *, bar, **parent_kwargs):
#             super().foo(*parent_args, **parent_kwargs)
#             ...
#
#     the synthesized function will roughly look like this:
#
#         def __synthetic__init__(self, a, b, c, /, *, foo, bar):
#             parent_args = (a, b)
#             parent_kwargs = dict(key=key)
#             func_args = (c,)
#             func_kwargs = dict(bar=bar)
#             wrapped_func(*parent_args, *func_args, **parent_kwargs, **func_kwargs)
#
#     Note:
#
#         - neither parent nor child func may have varargs.
#         -
#         - The child func may reuse keyword arguments from the parent func and give them different keywords.
#           - if keyword args are reused, they won't be included in parent_kwargs.
#         - additional positional only args must have defaults values (LSP!)
#         - additional positional only arguments are always added after positional-only arguments of the parent.
#     """
#     ...
