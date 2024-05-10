r"""Submodule containing general purpose decorators."""

__all__ = [
    # Functions
    "decorator",
    # Classes
    "DecoratorError",
]

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature

from typing_extensions import Any, Optional, Self, cast

from tsdm.types.aliases import BUILTIN_CONTAINERS, Nested
from tsdm.types.variables import Fun, P, R_co, T
from tsdm.utils.funcutils import rpartial

__logger__: logging.Logger = logging.getLogger(__name__)


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
                f", Optional: {param.default is EMPTY}"
                for key, param in sign.parameters.items()
            ),
            self.message,
        )
        return super().__str__() + "\n" + "\n".join(default_message)


# NOTE: Type hinting is severely limited because HKTs are not supported.
#   The lack of an explicit FunctionType is a problem.
def decorator(deco: Fun, /) -> Fun:
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

    Example:
        >>> def wrap_fun(
        ...     func: Callable,
        ...     before: Optional[Callable] = None,
        ...     after: Optional[Callable] = None,
        ...     /,
        ... ) -> Callable:
        ...     '''Wraps function.'''

    Here, there is a problem:

    >>> @wrap_func
    ... def func(x):
    ...     pass

    here, and also in the case of wrap_func(func), the result should be an identity operation.
    However, the other case

    >>> @wrap_func(before=lambda: print("Hello", end=""))
    ... def func():
    ...     print(" World!")
    >>> func()
    Hello World!

    the result is a wrapped function. The fundamental problem is a disambiguation between the cases.
    In either case, the decorator sees as input (callable, None, None) and so it cannot distinguish
    whether the first input is a wrapping, or the wrapped.

    Thus, we either need to abandon positional arguments with default values.

    Note however, that it is possible so save the situation by adding at least one
    mandatory positional argument:

    >>> def wrap_fun(
    ...     func: Callable,
    ...     before: Optional[Callable],
    ...     after: Optional[Callable] = None,
    ...     /,
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
    logger = __logger__.getChild(deco.__name__)
    logger.debug("Creating decorator.")

    deco_sig = signature(deco)
    ErrorHandler = DecoratorError(deco)

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

    @wraps(deco)
    def __parametrized_decorator(obj=None, /, *args, **kwargs):
        if obj is None:
            logger.debug(
                "Decorator used in BRACKET mode.\n"
                "Creating decorator with fixed arguments \n\targs=%s, \n\tkwargs=%s",
                args,
                kwargs,
            )
            return rpartial(deco, *args, **kwargs)

        logger.debug("Decorator used in FUNCTIONAL/BARE mode.")
        return deco(obj, *args, **kwargs)

    return cast(Fun, __parametrized_decorator)


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


def recurse_on_builtin_container(
    func: Callable[[T], R_co], /, *, kind: type[T]
) -> Callable[[Nested[T]], Nested[R_co]]:
    r"""Apply function to a nested iterables of a given kind.

    Args:
        kind: The type of the leave nodes
        func: A function to apply to all leave Nodes
    """
    if issubclass(kind, BUILTIN_CONTAINERS):  # type: ignore[misc, arg-type]
        raise TypeError(f"kind must not be a builtin container! Got {kind=}")

    @wraps(func)
    def recurse(x: Nested[T]) -> Nested[R_co]:
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
                return {recurse(obj) for obj in Set}  # pyright: ignore[reportUnhashable]
            case frozenset() as FrozenSet:
                return frozenset(recurse(obj) for obj in FrozenSet)
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
    raise NotImplementedError("Not yet implemented.")
