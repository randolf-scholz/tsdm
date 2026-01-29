r"""Base Classes for Encoders.

Some special encoders that can be considered are:

- inverse encoders: `~f(x) = f⁻¹(x)`
- chaining encoders: `(f @ g)(x) = g(f(x))`
- piping encoders: `(f >> g)(x) = f(g(x))`
- parallel encoders: `(f | g)((x, y)) = (f(x), g(y))`
- joint encoders: `(f & g)(x) = (f(x), g(x))`
  - inversion requires an aggregation function: `(f & g)⁻¹(y) = agg(f⁻¹(y), g⁻¹(y))`
  - in principle, one can use any element from the tuple.
  - but higher numerical precision can be achieved by aggregation.
  - default aggregation is to use a random element from the tuple.


Note on `BaseEncoder`:
- will wrap the `fit`, `encode`, and `decode` methods.
    - if the encoder requires fitting, encode/decode will raise an error if not fitted.
    - if the encoder does not require fitting,

Note:
    Golden Rule for implementation: init/fit can be slow, but transform should be fast.
    Use specialization/dispatch to ensure fast transforms.

Remark
======
Typing Encoders can be challenging, because not all types might be defined at
the time of instantiation. Often, the type of the encoder is only known after
fitting it to some data.

There are multiple ways to handle this:

1. Make fit return the encoder with the correct type.
   - Note that this would be a breaking change, since for instance sklearn encoders
     return `None` after fitting.
2. Add Manual `__new__` overloads that fall back to an upper bound type.
   - in the future, we can use the default type (PEP 696)
3. Use a polymorphic instead of a generic type.
   For example, we can have `StandardScalar(Encoder[NumericalArray, NumericalArray])`
   and then set `def encoder[Arr: NumericalArray](x: Arr) -> Arr: ...`

The use of polymorphic encoders also has effects on the chaining of encoders.
What is [T → T] >> [DataFrame → DataFrame]?

1. Should it be allowed?
2. If so, what is the type?

The only sensible answer to (2) is that it should be [DataFrame → DataFrame].
However it might be difficult for a type-checker to infer this correctly.
In principle, this would require some form of higher kinded types, then we could write an overload
of the form `(self: Poly[X], other: [X, Y]) -> Encoder[X, Y]: ...`.
Here, `Poly[T]` is a protocol describing a polymorphic encoder, with `X` being the upper bound.
That is, the encode signature is `Poly[T].encode[T: X](x: X) -> X: ...`.

Polymorphic encoders might be problematic for this very reason, and possibly should be avoided.



Remark: naming conventions:

- For ``OP(Encoder, Encoder)`` we use boolean/bitwise operators like `@`, `|`, `&`, `~`, `*`, and `**`.
- For ``OP(Enocder, int)`` we use integer operators like `*`, `**`, `%`, `//`.

# @: tensor product? (allows us to not having to define scalar multiplication separately?)
#    - nice equivalence to being compositon of linear maps.
#    So (f @ Tensor) is equivalent to `f >> LinearMap(Tensor)`.
# +: sum
# -: ?
# / num: ? reduce?
# % num: ? reduce?
# // num: ? concurrent/duplicate?
# ** num: repeat (>>)
# * num: duplicate? (⇝ similar to (x,) * n)
"""

__all__ = [
    # constants
    "ID",
    "CLONE",
    "WRAP_TUPLE",
    "UNWRAP_TUPLE",
    # protocols & abcs
    "BaseEncoder",
    "Encoder",
    "EncoderDict",
    "EncoderList",
    "EncoderMeta",
    "EncoderProtocol",
    "FittableEncoder",
    "ParametrizedEncoder",
    "StaticEncoder",
    # classes
    "DeepcopyEncoder",
    "IdentityEncoder",
    "MappedEncoder",
    "NestedEncoder",
    "TupleUnwrapper",
    "TupleWrapper",
    "WrappedEncoder",
    # functions
    "map_encoders",
    "nest_encoder",
    "simplify",
    "wrap",
    # ALGEBRA SUBMODULE
    # classes
    "Choice",
    "Compose",
    "Diagonal",
    "Duplicate",  # %
    "Fold",
    "Fork",  # &
    "InverseEncoder",  # ~
    "Meet",
    "Parallel",
    "Pipe",  # >>
    "Repeat",  # **
    "Replicate",
    # functions
    "choice",
    "compose",
    "diagonal",
    "duplicate",
    "fold",
    "fork",  # &
    "invert",  # ~
    "meet",
    "parallel",
    "pipe",  # >>
    "repeat",  # **
    "replicate",
]

import inspect
import logging
import pickle
import random
from abc import abstractmethod
from collections.abc import Callable as Fn, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, fields, is_dataclass
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Final,
    Literal as L,
    Optional,
    Protocol,
    Self,
    _ProtocolMeta as ProtocolMeta,
    assert_type,
    cast,
    final,
    overload,
    runtime_checkable,
)
from warnings import deprecated

from tsdm.constants import UNDEFINED
from tsdm.encoders.protocols import (
    Reduction,
    SupportsDecode,
    SupportsEncode,
    SupportsFit,
    SupportSimplify,
)
from tsdm.types.aliases import DictArg, FilePath, NestedBuiltin
from tsdm.types.utils import is_classvar
from tsdm.utils.decorators import pprint_mapping, pprint_repr, pprint_sequence
from tsdm.utils.funcutils import recurse_on_nested_builtin


@runtime_checkable
class EncoderProtocol[X, Y](Protocol):
    r"""Minimal Protocol for Encoders.

    This protocol should be used in applications that only use encoders, but do not need to
    worry about creating new encoders or chaining them together.
    """

    @abstractmethod
    def fit(self, x: X, /) -> None: ...
    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...


class Encoder[X, Y](Protocol):
    r"""Protocol for Encoders with algebraic mixin methods."""

    # region abstract methods ----------------------------------------------------------
    @property
    @abstractmethod
    def params(self) -> Mapping[str, Any]: ...

    @property
    @abstractmethod
    def requires_fit(self) -> bool: ...

    @abstractmethod
    def fit(self, x: X, /) -> None: ...
    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...

    # endregion abstract methods -------------------------------------------------------

    # @property
    # def required_params(self) -> frozenset[str]:
    #     r"""The required parameters to initialize the encoder."""
    #     return frozenset(self.params)

    # def simplify(self) -> "Encoder[X, Y]":
    #     r"""Simplify the encoder."""
    #     return self

    # region scikit-learn compatibility ------------------------------------------------
    # def fit_transform(self, x: X, /) -> Y:
    #     r"""Fit the encoder to the data and apply the transformation."""
    #     self.fit(x)
    #     return self.encode(x)
    #
    # def transform(self, x: X, /) -> Y:
    #     r"""Alias for encode."""
    #     return self.encode(x)
    #
    # def inverse_transform(self, y: Y, /) -> X:
    #     r"""Alias for decode."""
    #     return self.decode(y)
    #

    #
    # endregion scikit-learn compatibility ---------------------------------------------

    # region magic methods -------------------------------------------------------------
    # NOTE: We exclude the magic methods from the protocol, because diverging
    #   protocols raise recursion error in mypy. They might be added later.
    # FIXME: https://github.com/python/mypy/issues/17326
    # endregion magic methods ----------------------------------------------------------


class ParametrizedEncoder[X, Y](EncoderProtocol[X, Y], Protocol):
    r"""Protocol for encoders with parameters."""

    @property
    @abstractmethod
    def required_params(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]: ...

    @abstractmethod
    def set_params(self, mapping: Mapping[str, Any], /, **kwargs: Any) -> None: ...

    # region mixin methods -------------------------------------------------------------
    @classmethod
    def from_params(cls, mapping: Mapping[str, Any], /, **kwargs: Any) -> Self:
        r"""Create an encoder from parameters."""
        options = dict(mapping, **kwargs)
        obj = object.__new__(cls)
        obj.set_params(options)
        return obj

    @property
    def requires_fit(self) -> bool:
        r"""Check if the encoder requires fitting."""
        params = self.params
        return any(params[key] is UNDEFINED for key in self.required_params)

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        r"""Alias for `self.params`."""
        return self.params

    # endregion mixin methods ---------------------------------------------------------


class EncoderMeta(ProtocolMeta):
    r"""Metaclass for Encoders."""

    LOGGER: logging.Logger = logging.getLogger(__name__)

    @property
    def FIELDS(cls) -> frozenset[str]:  # noqa: N802
        r"""Fields that are considered for the encoder."""
        if is_dataclass(cls):
            return frozenset({f.name for f in fields(cls)})

        # Fallback: inspect the annotations of the class.
        annotations = getattr(cls, "__annotations__", {})
        return frozenset(
            key for key, val in annotations.items() if not is_classvar(val)
        )

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> type:
        r"""Create a new Encoder class."""
        if "FIELDS" not in namespace and not inspect.isabstract(cls):
            # use the default, but users can override it.
            namespace["FIELDS"] = cls.FIELDS

        new_type = super().__new__(cls, name, bases, namespace, **kwds)
        new_type.__annotations__["FIELDS"] = ClassVar[frozenset[str]]
        return new_type


class BaseEncoder[X, Y](Encoder[X, Y], metaclass=EncoderMeta):
    r"""Abstract base class for encoders."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the Encoder."""

    @final
    def __call__(self, x: X, /) -> Y:
        return self.encode(x)

    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...
    @abstractmethod
    def fit(self, x: X, /) -> None: ...

    # region hook interface ------------------------------------------------------------
    # we offer pre- and post- hooks for fit, encode, and decode methods.
    # as well as a general post-init hook
    # hooks must be registered on the class-level as lists of callables.
    post_init_hooks: ClassVar[list[Fn[[Self], None]]] = []  # type: ignore[misc]
    pre_fit_hooks: ClassVar[list[Fn[[Self], None]]] = []  # type: ignore[misc]
    post_fit_hooks: ClassVar[list[Fn[[Self], None]]] = []  # type: ignore[misc]
    pre_encode_hooks: ClassVar[list[Fn[[Self], None]]] = []  # type: ignore[misc]
    post_encode_hooks: ClassVar[list[Fn[[Self], None]]] = []  # type: ignore[misc]
    pre_decode_hooks: ClassVar[list[Fn[[Self], None]]] = []  # type: ignore[misc]
    post_decode_hooks: ClassVar[list[Fn[[Self], None]]] = []  # type: ignore[misc]
    pre_hooks: ClassVar[dict[str, list[Fn[[Self], None]]]] = {}  # type: ignore[misc]
    post_hooks: ClassVar[dict[str, list[Fn[[Self], None]]]] = {}  # type: ignore[misc]

    def __init_subclass__(cls) -> None:
        r"""Initialize subclass hooks."""
        super().__init_subclass__()
        cls.encode = cls.with_hooks(cls.encode)
        cls.decode = cls.with_hooks(cls.decode)
        cls.fit = cls.with_hooks(cls.fit)
        cls.pre_hooks["fit"] = cls.pre_fit_hooks
        cls.post_hooks["fit"] = cls.post_fit_hooks
        cls.pre_hooks["encode"] = cls.pre_encode_hooks
        cls.post_hooks["encode"] = cls.post_encode_hooks
        cls.pre_hooks["decode"] = cls.pre_decode_hooks
        cls.post_hooks["decode"] = cls.post_decode_hooks

    @staticmethod
    def with_hooks[T: BaseEncoder, **P, R](
        method: Fn[Concatenate[T, P], R],
    ) -> Fn[Concatenate[T, P], R]:
        r"""Decorator to add hooks to methods."""

        @wraps(method)
        def wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
            for hook in self.pre_hooks[method.__name__]:
                hook(self)
            result = method(self, *args, **kwargs)
            for hook in self.post_hooks[method.__name__]:
                hook(self)
            return result

        return wrapper

    # endregion hook interface ---------------------------------------------------------

    # region serialization interface ---------------------------------------------------
    def is_serializable(self) -> bool:
        r"""Check if the encoder is serializable."""
        params = self.params
        return not any(params[key] is UNDEFINED for key in self.params)

    def serialize(self, filepath: FilePath, /) -> None:
        r"""Serialize the encoder to a file."""
        if not self.is_serializable():
            raise RuntimeError("Encoder is not serializable!")

        with Path(filepath).open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def deserialize(cls, filepath: FilePath, /) -> Self:
        r"""Deserialize the encoder from a file."""
        with open(filepath, "rb") as file:
            obj = pickle.load(file)
            if not isinstance(obj, cls):
                raise TypeError(f"Deserialized object is not an instance of {cls}.")
        return obj

    # endregion serialization interface ------------------------------------------------

    # region simplify interface --------------------------------------------------------
    def simplify(self) -> BaseEncoder[X, Y]:
        r"""Simplify the encoder."""
        return self

    # endregion simplify interface -----------------------------------------------------

    # region parameter interface -------------------------------------------------------
    FIELDS: ClassVar[frozenset[str]]
    r"""Fields that are considered for the encoder."""

    @property
    def params(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.FIELDS}

    @cached_property
    def requires_fit(self) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""Check if the encoder requires fitting."""
        return any(
            (val is UNDEFINED or getattr(val, "requires_fit", False))
            for val in self.params.values()
        )

    def validate_params(self) -> None:
        r"""Validate the encoder parameters.

        Automatically called after fitting the encoder.
        """
        errors: list[Exception] = []

        for key in self.FIELDS:
            val = getattr(self, key)
            if val is UNDEFINED:
                msg = f"Parameter '{key}' is not defined."
                errors.append(AssertionError(msg))
            if getattr(val, "requires_fit", False):
                msg = f"Parameter '{key}' still requires fitting."
                errors.append(AssertionError(msg))

        if errors:
            raise ExceptionGroup("Parameter validation failed", errors)

    # endregion parameter interface ----------------------------------------------------

    # region magic methods -------------------------------------------------------------
    def __invert__(self) -> BaseEncoder[Y, X]:
        r"""Return the inverse encoder (i.e. decoder).

        See Also: `InverseEncoder`
        """
        return invert(self)

    # region SISO ----------------------------------------------------------------------
    def __rshift__[Z](self, other: Encoder[Y, Z], /) -> Pipe[X, Z]:
        r"""Apply encoders in order (``>>``).

            x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y

        See Also: `Pipe`, `pipe`
        """
        return pipe(self, other)

    def __rrshift__[T](self, other: Encoder[T, X], /) -> Pipe[T, Y]:
        r"""Apply encoders in order (``>>``).

            x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y

        See Also: `Pipe`, `pipe`
        """
        return pipe(other, self)

    def __pow__[T](self: Encoder[T, T], num: int, /) -> Repeat[T]:
        r"""Repeat an encoder n times (``**``).

            x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)

        See Also: `Repeat`, `repeat`
        """
        return repeat(self, num)

    # endregion SISO -------------------------------------------------------------------

    # region MIMO ----------------------------------------------------------------------
    def __xor__[X2, Y2](self, other: Encoder[X2, Y2], /) -> Parallel[tuple[X, X2], tuple[Y, Y2]]:  # fmt: skip
        r"""Return product encoders.

            x₁ ───▶ f₁(x₁)
            x₂ ───▶ f₂(x₂)
            ⋮         ⋮
            xₙ ───▶ fₙ(xₙ)

        See Also: `Parallel`
        """
        return parallel(self, other)

    def __rxor__[X2, Y2](self, other: Encoder[X2, Y2], /) -> Parallel[tuple[X2, X], tuple[Y2, Y]]:  # fmt: skip
        r"""Return product encoders.

            x₁ ───▶ f₁(x₁)
            x₂ ───▶ f₂(x₂)
            ⋮         ⋮
            xₙ ───▶ fₙ(xₙ)

        See Also: `Parallel`
        """
        return parallel(other, self)

    def __mod__(self, num: int, /) -> Parallel[tuple[X, ...], tuple[Y, ...]]:
        r"""Fork the encoder into multiple encoders (``*``).

            x₁ ───▶ f(x₁)
            x₂ ───▶ f(x₂)
            ⋮        ⋮
            xₙ ───▶ f(xₙ)

        See Also: `Replicate`, `Parallel`
        """
        return replicate(self, num)

    # endregion MIMO -------------------------------------------------------------------

    # region SIMO ----------------------------------------------------------------------
    # TODO: automatically combine `e1 & e2 & e3` into a single meet?
    def __or__[Y2](self, other: Encoder[X, Y2], /) -> Fork[X, tuple[Y, Y2]]:
        r"""Execute multiple encoders with the same input (``&``).

                  ┌────▶ f₁(x)
            x ────┼────▶ f₂(x)
                  │        ⋮
                  └────▶ fₙ(x)

        See Also: `Fork`
        """
        # FIXME: mypy does not predict correct return type...
        result = fork(self, other)
        assert_type(result, Fork[X, tuple[Y, Y2]])
        return result

    # TODO: automatically combine `e1 & e2 & e3` into a single meet?
    def __ror__[Y2](self, other: Encoder[X, Y2], /) -> Fork[X, tuple[Y2, Y]]:
        r"""Execute multiple encoders with the same input (``&``).

                  ┌────▶ f₁(x)
            x ────┼────▶ f₂(x)
                  │        ⋮
                  └────▶ fₙ(x)

        See Also: `Fork`
        """
        # FIXME: mypy does not predict correct return type...
        result = fork(other, self)
        assert_type(result, Fork[X, tuple[Y2, Y]])
        return result

    def __mul__(self, other: int, /) -> Fork[X, tuple[Y, ...]]:
        r"""Duplicate the encoder multiple times (``%``).

                  ┌────▶ f(x)
            x ────┼────▶ f(x)
                  │        ⋮
                  └────▶ f(x)

        See Also: `Duplicate`, `Fork`
        """
        return duplicate(self, other)

    # endregion SIMO -------------------------------------------------------------------

    # region MISO ----------------------------------------------------------------------
    def __and__[X2](self, other: Encoder[X2, Y], /) -> Meet[tuple[X, X2], Y]:
        r"""Combine two encoders into a single encoder (``&``).

            x₁ ────┐
            x₂ ────┼────▶ choice([f₁(x₁), f₂(x₂), ..., fₙ(xₙ)])
            ⋮      │
            xₙ ────┘

        See Also: `Meet`
        """
        return meet(self, other)

    def __rand__[X2](self, other: Encoder[X2, Y], /) -> Meet[tuple[X2, X], Y]:
        r"""Combine two encoders into a single encoder (``&``).

            x₁ ────┐
            x₂ ────┼────▶ choice([f₁(x₁), f₂(x₂), ..., fₙ(xₙ)])
            ⋮      │
            xₙ ────┘

        See Also: `Meet`
        """
        return meet(other, self)

    # TODO: possibly better to use reduction rather than integer RHS.
    def __floordiv__(self, other: int, /) -> Meet[tuple[X, ...], Y]:
        r"""Aggregate multiple outputs via reduction  (``//``).

            x₁ ────┐
            x₂ ────┼────▶ choice([f₁(x₁), f₂(x₂), ..., fₙ(xₙ)])
             ⋮     │
            xₙ ────┘

        See Also: `Meet`
        """
        return fold(self, other)

    # endregion MISO -------------------------------------------------------------------

    # region arithmetic methods --------------------------------------------------------
    def __add__(self, other: Any, /) -> Any:
        r"""Add two encoders together (``+``).

            x ───▶ f(x) + g(x)

        See Also: `Sum`, `add`
        """
        raise NotImplementedError

    def __radd__(self, other: Any, /) -> Any:
        r"""Add two encoders together (``+``).

            x ───▶ f(x) + g(x)

        See Also: `Sum`, `add`
        """
        raise NotImplementedError

    def __sub__(self, other: Any, /) -> Any:
        r"""Subtract two encoders (``-``).

            x ───▶ f(x) - g(x)

        See Also: `Difference`, `subtract`
        """
        raise NotImplementedError

    def __rsub__(self, other: Any, /) -> Any:
        r"""Subtract two encoders (``-``).

            x ───▶ f(x) - g(x)

        See Also: `Difference`, `subtract`
        """
        raise NotImplementedError

    def __matmul__(self, other: Any, /) -> Pipe:
        r"""Apply reducing tensor contraction (``@``) (MISO).

            f @ Tensor  ≝  f >> LinearMap(Tensor, upper_indices="ALL")
            f @ (Tensor, sig)  ≝  f >> LinearMap(Tensor, sig)

        Note:
            Scalar multiplication can be realized using rank-0 tensors.

            (2 @ f) ≝ x ───▶ 2x ───▶ f(2x)
            (f @ 2) ≝ x ───▶ f(x) ───▶ 2f(x)

        See Also : `LinearMap`, `linear_map`, `Pipe`
        """
        raise NotImplementedError

    def __rmatmul__(self, other: Any, /) -> Pipe:
        r"""Apply tensor contraction (``@``).

            Tensor @ f  ≝  LinearMap(Tensor, lower_indices="ALL") >> f
            (Tensor, sig) @ f  ≝  LinearMap(Tensor, sig) >> f

        See Also : `LinearMap`, `linear_map`, `Pipe`
        """
        raise NotImplementedError

    # endregion magic methods ----------------------------------------------------------
    # endregion magic methods ----------------------------------------------------------

    # region fluent interface ----------------------------------------------------------
    # region magic methods -------------------------------------------------------------
    def pipe[Z](self, other: Encoder[Y, Z], /) -> Pipe[X, Z]:
        r"""Chain the encoder with another encoder (``>>``).

        See Also: `Pipe`, `__rshift__`
        """
        return self >> other

    def repeat[T](self: Encoder[T, T], num: int, /) -> Repeat[T]:
        r"""Repeat the encoder multiple times (``**``).

        See Also: `Repeat`, `__pow__`
        """
        return repeat(self, num)

    def parallel[X2, Y2](self, other: Encoder[X2, Y2], /) -> Parallel[tuple[X, X2], tuple[Y, Y2]]:  # fmt: skip
        r"""Combine two encoders into a single encoder (``^``).

        See Also: `Parallel`, `__xor__`
        """
        return parallel(self, other)

    def replicate(self, num: int, /) -> Parallel[tuple[X, ...], tuple[Y, ...]]:
        r"""Replicate the encoder multiple times (``%``).

        See Also: `Replicate`, `__mod__`
        """
        return replicate(self, num)

    def fork[Y2](self, other: Encoder[X, Y2], /) -> Fork[X, tuple[Y, Y2]]:
        r"""Execute multiple encoders with the same input (``|``).

        See Also: `Fork`, `__or__`
        """
        return fork(self, other)

    def duplicate(self, num: int, /) -> Fork[X, tuple[Y, ...]]:
        r"""Duplicate the encoder multiple times (``*``).

        See Also: `Duplicate`, `Fork`, `__mul__`
        """
        return duplicate(self, num)

    def meet[X2](self, other: Encoder[X2, Y], /) -> Meet[tuple[X, X2], Y]:
        r"""Combine two encoders into a single encoder (``&``).

        See Also: `Meet`, `__and__`
        """
        return meet(self, other)

    def fold(self, num: int, /) -> Meet[tuple[X, ...], Y]:
        r"""Aggregate multiple outputs via reduction (``//``).

        See Also: `Fold`, `__floordiv__`
        """
        return fold(self, num)

    # endregion magic methods ----------------------------------------------------------

    # region other methods -------------------------------------------------------------
    def standardize(self) -> BaseEncoder[X, Y]:
        r"""Chain a standardizer."""
        import tsdm.encoders as E

        return self >> E.StandardScaler()

    def minmax_scale(self) -> BaseEncoder[X, Y]:
        r"""Chain a minmax scaling."""
        import tsdm.encoders as E

        return self >> E.MinMaxScaler()

    # endregion other methods ----------------------------------------------------------
    # endregion fluent interface -------------------------------------------------------


# fmt: off
@overload
def simplify[X, Y](e: BaseEncoder[X, Y], /) -> BaseEncoder[X, Y]: ...
@overload
def simplify[X, Y](e: Encoder[X, Y], /) -> Encoder[X, Y]: ...
# fmt: on
def simplify[X, Y](encoder: Encoder[X, Y], /) -> Encoder[X, Y]:  # fmt: skip
    r"""Simplify the encoder.

    This will call the `simplify` method of the encoder if it exists.
    Otherwise, it will return the encoder as is.
    """
    if isinstance(encoder, SupportSimplify):
        return encoder.simplify()
    return wrap(encoder)


class FittableEncoder[X, Y](BaseEncoder[X, Y]):
    r"""Base class for encoders implemented within this package."""

    # region abstract methods ----------------------------------------------------------
    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...
    @abstractmethod
    def fit(self, x: X, /) -> None: ...

    # endregion abstract methods -------------------------------------------------------

    def __setattr__(self, key: str, value: object, /) -> None:
        if key in self.FIELDS:
            with suppress(AttributeError):
                del self.requires_fit  # clear requires_fit flag
        super().__setattr__(key, value)

    def assert_fitted(self, /) -> None:
        r"""Assert that the encoder has been fitted."""
        if self.requires_fit:
            raise AssertionError("Encoder has not been fitted!")

    # NOTE: not putting annotation on these makes mypy incredibly slow.
    pre_encode_hooks: ClassVar[list[Fn[..., None]]] = [assert_fitted]
    pre_decode_hooks: ClassVar[list[Fn[..., None]]] = [assert_fitted]
    post_fit_hooks: ClassVar[list[Fn[..., None]]] = [
        BaseEncoder.validate_params,
        assert_fitted,
    ]


class StaticEncoder[X, Y](BaseEncoder[X, Y]):
    r"""An encoder that never requires fitting.

    Note that instances of this class still can have parameters, but they have to be
    provided/determined at initialization time.
    """

    requires_fit: Final[L[False]] = False  # pyright: ignore[reportIncompatibleVariableOverride]  # noqa: PYI064
    post_fit_hooks = [BaseEncoder.validate_params]

    @final
    def fit(self, _: X, /) -> None:
        r"""Noop since the encoder is static."""

    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...


@pprint_sequence(recursive=2)
class EncoderList[
    X,  # invariant
    Y,  # invariant
    E: Encoder,  # covariant
](FittableEncoder[X, Y], Sequence[E]):
    r"""Wraps a list of encoders.

    Such objects should consist of 3 parts of data:

    1. The sequence of encoders.
    2. Dependent extra data, that can be derived from the encoders.
    3. Independent extra data.
    """

    @classmethod
    def new[E2: Encoder](
        cls: type[EncoderList], *, encoders: Iterable[E2]
    ) -> EncoderList[Any, Any, E2]:
        r"""Create a new instance with the given values."""
        try:
            result = cls(encoders)
        except TypeError as exc:
            exc.add_note(
                f"Failed to create {cls.__name__} with encoders={encoders!r}."
                f" Possibly the `new` method is not implemented correctly."
            )
            raise
        if type(result) is not cls:
            raise TypeError(
                f"Creating {cls.__name__} produced an unexpected type."
                f"\n\tGot {type(result)}, expected {cls.__name__}."
                f"\n\tThis can happen if {cls.__name__} does not correctly override `new`."
            )
        return result

    @property
    def encoders(self) -> Sequence[E]:
        r"""The raw sequence of encoders."""
        return self._encoders

    @cached_property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self)

    #  region abstract implementation --------------------------------------------------
    def __init__(self, encoders: Iterable[E] = (), /) -> None:
        self._encoders: Final[Sequence[E]] = [wrap(e) for e in encoders]  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]

    def __len__(self) -> int:
        return len(self._encoders)

    @overload
    def __getitem__(self, index: int, /) -> E: ...
    @overload
    def __getitem__(self, index: slice, /) -> EncoderList[Any, Any, E]: ...
    def __getitem__(self, index: int | slice, /) -> E | EncoderList[Any, Any, E]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(index, slice):
            result = self.new(encoders=self._encoders[index])
            if type(result) is not type(self):
                raise TypeError(
                    f"Slicing produced an unexpected type."
                    f"\n\tGot {type(result)}, expected {type(self)}."
                    f"\n\tThis can happen if {type(self)} does not correctly override `new`."
                )
            return result
        return self._encoders[index]

    # endregion abstract implementation ------------------------------------------------


@pprint_mapping(recursive=2)
@dataclass(init=False)
class EncoderDict[
    X,  # invariant
    Y,  # invariant
    K,  # invariant
    E: Encoder = Encoder,  # covariant
](FittableEncoder[X, Y], Mapping[K, E]):
    r"""Wraps dictionary of encoders."""

    @classmethod
    @abstractmethod
    def new(cls, *, encoders: DictArg) -> EncoderDict: ...
    @abstractmethod
    def encode(self, x: X, /) -> Y: ...
    @abstractmethod
    def decode(self, y: Y, /) -> X: ...
    @abstractmethod
    def fit(self, x: X, /) -> None: ...

    # region abstract implementation ---------------------------------------------------
    def __init__[K2, E2: Encoder](
        self: EncoderDict[Any, Any, K2, E2],
        encs: DictArg[K2, E2] = (),
        /,
    ) -> None:
        self._encoders: Final[Mapping[K, E]] = dict(encs)

    # fmt: off
    def __len__(self) -> int: return len(self._encoders)
    def __iter__(self) -> Iterator[K]: return iter(self._encoders)
    def __getitem__(self, key: K, /) -> E: return self._encoders[key]
    # fmt: on
    # endregion abstract implementation ------------------------------------------------

    @property
    def encoders(self) -> Mapping[K, E]:
        return self._encoders

    @cached_property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self.values())

    def simplify(self) -> EncoderDict[X, Y, K, Encoder]:
        return self.new(encoders={k: simplify(e) for k, e in self.items()})


@pprint_repr
@dataclass
class WrappedEncoder[X, Y](FittableEncoder[X, Y]):
    r"""Wraps an `Encoder` to a `BaseEncoder`."""

    FIELDS: ClassVar[frozenset[str]] = frozenset({"encoder", "decoder"})
    r"""The names of the parameters of the encoder."""

    encoder: SupportsEncode[X, Y] | Fn[[X], Y] | None = None
    r"""The encoder/function to wrap."""
    decoder: SupportsEncode[Y, X] | Fn[[Y], X] | None = None
    r"""The decoder/function to wrap, if available."""

    def __invert__(self) -> BaseEncoder[Y, X]:
        return wrap(encoder=self.decoder, decoder=self.encoder)

    def __post_init__(self) -> None:
        match self.encoder, self.decoder:
            # easy cases
            case None, None:
                raise ValueError(
                    "At least one of `encoder` or `decoder` must be provided."
                )

            # only one of them provided
            case e, None:
                assert e is not None
                self.encode = e.encode if isinstance(e, SupportsEncode) else e
                if isinstance(e, SupportsDecode):
                    self.decode = e.decode  # type: ignore[unreachable]
            case None, d:
                assert d is not None
                self.decode = d.encode if isinstance(d, SupportsEncode) else d
                if isinstance(d, SupportsDecode):
                    self.encode = d.decode  # type: ignore[unreachable]

            # ambiguous cases
            case SupportsEncode() as e, SupportsEncode() as d:
                self.encode = e.encode
                self.decode = d.encode
            case SupportsEncode() as e, d if callable(d):
                self.encode = e.encode
                self.decode = d
            case e, SupportsEncode() as d if callable(e):
                self.encode = e
                self.decode = d.encode
            case e, d if callable(e) and callable(d):
                self.encode = e
                self.decode = d
            case _ as never:
                raise TypeError(f"Unsupported encoder/decoder types: {never}")

    @property
    def params(self) -> dict[str, Any]:
        return getattr(self.encoder, "params", {})

    def fit(self, x: X, /) -> None:
        r"""Fit the encoder if it is a `FittableEncoder`."""
        match self.encoder:
            case SupportsFit() as fittable:
                fittable.fit(x)  # type: ignore[unreachable]
            case _:
                pass

    def encode(self, x: X, /) -> Y:
        # overwritten in __post_init__
        raise NotImplementedError

    def decode(self, y: Y, /) -> X:
        # overwritten in __post_init__
        raise NotImplementedError

    def simplify(self) -> BaseEncoder[X, Y]:
        if (simplify := getattr(self.encoder, "simplify", None)) is not None:
            e = simplify()
            return e if isinstance(e, FittableEncoder) else WrappedEncoder(e)
        return self.encoder if isinstance(self.encoder, FittableEncoder) else self


# fmt: off
@overload  # yield BaseEncoder as-is
def wrap[T: FittableEncoder](encoder: T, /) -> T: ...
@overload
def wrap[X=Any, Y=Any](
    encoder: SupportsEncode[X, Y] | Fn[[X], Y] | None = ...,  # pyright: ignore[reportInvalidTypeVarUse]
    decoder: SupportsEncode[Y, X] | Fn[[Y], X] | None = ...,
) -> BaseEncoder[X, Y]: ...
# fmt: on
def wrap[X=Any, Y=Any](
    encoder: SupportsEncode[X, Y] | Fn[[X], Y] | None = None,
    decoder: SupportsEncode[Y, X] | Fn[[Y], X] | None = None,
) -> BaseEncoder[X, Y]:  # fmt: skip
    r"""Wrap a (pair of) function as an encoder.

    This will create a `WrappedEncoder` that calls the function `fn` on the input data.
    """
    if isinstance(encoder, FittableEncoder):  # return as-is
        return encoder
    return WrappedEncoder(encoder, decoder)


class MappedEncoder[
    MappingIn: Mapping[str, Any],  # Mapping[K, X]
    MappingOut: Mapping[str, Any],  # Mapping[K, Y]
](EncoderDict[MappingIn, MappingOut, str, Encoder]):  # Encoder[X, Y]
    r"""Maps encoders to keys.

        (k₁, x₂) ────▶ f_{k₁}(x₁)
        (k₂, x₂) ────▶ f_{k₂}(x₂)
            ⋮              ⋮
        (kₙ, xₙ) ────▶ f_{kₙ}(xₙ)

    Example:
        >>> from tsdm.encoders import MappedEncoder, wrap
        >>> e1 = wrap(lambda x: f"{x} + 1")
        >>> e2 = wrap(lambda x: f"2 * {x}")
        >>> enc = MappedEncoder({"key1": e1, "key2": e2})
        >>> assert enc({"key1": "a", "key2": "b"}) == {"key1": "a + 1", "key2": "2 * b"}
    """

    @classmethod
    def new[X, Y](
        cls, *, encoders: DictArg[str, Encoder[X, Y]]
    ) -> MappedEncoder[Mapping[str, X], Mapping[str, Y]]:
        return MappedEncoder(encoders)

    @overload
    def __init__[X, Y](
        self: MappedEncoder[Mapping[str, X], Mapping[str, Y]],
        encoders: DictArg[str, Encoder[X, Y]] = ...,
        /,
    ) -> None: ...
    @overload
    def __init__(self, encoders: DictArg[str, Encoder] = ..., /) -> None: ...
    def __init__(self, encoders: DictArg[str, Encoder] = (), /) -> None:
        super().__init__(encoders)

    def __invert__(self) -> MappedEncoder[MappingOut, MappingIn]:
        # FIXME: https://github.com/python/typing/issues/548
        return cast(
            "MappedEncoder[MappingOut, MappingIn]",
            map_encoders({k: invert(e) for k, e in self.items()}),
        )

    def fit(self, xmap: MappingIn, /) -> None:
        if missing_keys := self.keys() - xmap.keys():
            raise ValueError(f"No data to fit encoders {missing_keys}.")
        if extra_keys := xmap.keys() - self.keys():
            raise ValueError(f"Extra data with no matching encoder {extra_keys}.")

        for k, x in xmap.items():
            self[k].fit(x)

    def encode(self, xmap: MappingIn, /) -> MappingOut:
        ymap = {k: self[k].encode(x) for k, x in xmap.items()}
        return cast("MappingOut", ymap)

    def decode(self, ymap: MappingOut, /) -> MappingIn:
        xmap = {k: self[k].decode(y) for k, y in ymap.items()}
        return cast("MappingIn", xmap)

    def simplify(self) -> MappedEncoder[MappingIn, MappingOut]:
        r"""Simplify the encoders."""
        return MappedEncoder[MappingIn, MappingOut](super().simplify())  # type: ignore[return-value]


def map_encoders[X, Y](
    encoders: Mapping[str, Encoder[X, Y]], /
) -> MappedEncoder[Mapping[str, X], Mapping[str, Y]]:
    r"""Map encoders.

        (k₁, x₂) ────▶ f_{k₁}(x₁)
        (k₂, x₂) ────▶ f_{k₂}(x₂)
                   ⋮
        (kₙ, xₙ) ────▶ f_{kₙ}(xₙ)

    See Also: `MappedEncoder`
    """
    return MappedEncoder(encoders)


@pprint_repr
@dataclass
class NestedEncoder[X, Y](FittableEncoder[NestedBuiltin[X], NestedBuiltin[Y]]):
    r"""Apply an encoder recursively to nested data structure.

    Any instances of the leaf type will be encoded using the encoder.
    Containers in the standard library will be recursed into
    (applies to `list`, `tuple`, `dict`, `set` and `frozenset`).
    Other types will raise `TypeError`.

    TODO: add support to pass other types as-is.
    """

    encoder: Encoder[X, Y]
    r"""The encoder to apply nested."""

    _: KW_ONLY

    leaf_type: type[X]
    r"""The type of the leaf elements."""
    output_leaf_type: type[Y]
    r"""The type of the output elements."""

    # FIXME: https://github.com/python/typing/issues/548
    def __invert__(self) -> NestedEncoder[Y, X]:
        return nest_encoder(
            invert(self.encoder),
            leaf_type=self.output_leaf_type,
            output_leaf_type=self.leaf_type,
        )

    def fit(self, x: NestedBuiltin[X], /) -> None:
        pass

    def encode(self, x: NestedBuiltin[X], /) -> NestedBuiltin[Y]:
        return recurse_on_nested_builtin(
            x,
            leaf_type=self.leaf_type,
            leaf_fn=self.encoder.encode,
        )

    def decode(self, y: NestedBuiltin[Y], /) -> NestedBuiltin[X]:
        return recurse_on_nested_builtin(
            y,
            leaf_type=self.output_leaf_type,
            leaf_fn=self.encoder.decode,
        )

    def simplify(self) -> NestedEncoder[X, Y]:
        return NestedEncoder(
            simplify(self.encoder),
            leaf_type=self.leaf_type,
            output_leaf_type=self.output_leaf_type,
        )


def nest_encoder[X, Y](
    encoder: Encoder[X, Y], /, *, leaf_type: type[X], output_leaf_type: type[Y]
) -> NestedEncoder[X, Y]:
    r"""Create a nested encoder that applies the given encoder recursively."""
    return NestedEncoder(
        encoder, leaf_type=leaf_type, output_leaf_type=output_leaf_type
    )


########################################################################################
# STATIC ENCODERS / CANONICAL INSTANCES                                                #
########################################################################################


# region static encoders ---------------------------------------------------------------
class IdentityEncoder(StaticEncoder[Any, Any]):
    r"""Identity function as an encoder."""

    FIELDS: ClassVar[frozenset[str]] = frozenset()
    r"""The names of the parameters of the encoder."""

    def __invert__(self) -> Self:
        return self

    def encode[T](self, x: T, /) -> T:
        return x

    def decode[T](self, y: T, /) -> T:
        return y

    def simplify(self) -> Self:
        return self


class DeepcopyEncoder(StaticEncoder[Any, Any]):
    r"""Encoder that deepcopies the input."""

    FIELDS: ClassVar[frozenset[str]] = frozenset()
    r"""The names of the parameters of the encoder."""

    def __invert__(self) -> Self:
        return self

    def encode[T](self, x: T, /) -> T:
        return deepcopy(x)

    def decode[T](self, y: T, /) -> T:
        return deepcopy(y)

    def simplify(self) -> Self:
        return self


class TupleWrapper(StaticEncoder[Any, tuple[Any]]):
    r"""Wraps input into a tuple."""

    FIELDS: ClassVar[frozenset[str]] = frozenset()
    r"""The names of the parameters of the encoder."""

    def __invert__(self) -> TupleUnwrapper:
        return TupleUnwrapper()

    def encode[T](self, x: T, /) -> tuple[T]:
        return (x,)

    def decode[T](self, y: tuple[T], /) -> T:
        return y[0]

    def simplify(self) -> Self:
        return self


class TupleUnwrapper(StaticEncoder[tuple[Any], Any]):
    r"""Unwraps input from a tuple."""

    FIELDS: ClassVar[frozenset[str]] = frozenset()
    r"""The names of the parameters of the encoder."""

    def __invert__(self) -> TupleWrapper:
        return TupleWrapper()

    def encode[T](self, y: tuple[T], /) -> T:
        return y[0]

    def decode[T](self, x: T, /) -> tuple[T]:
        return (x,)

    def simplify(self) -> Self:
        return self


ID: Final[IdentityEncoder] = IdentityEncoder()
r"""Canonical identity encoder."""
CLONE: Final[DeepcopyEncoder] = DeepcopyEncoder()
r"""Canonical deepcopy encoder."""
WRAP_TUPLE: Final[TupleWrapper] = TupleWrapper()
r"""Canonical tuple encoder."""
UNWRAP_TUPLE: Final[TupleUnwrapper] = TupleUnwrapper()
r"""Canonical tuple decoder."""


########################################################################################
# region ALGEBRAIC INTERFACE                                                           #
########################################################################################


@dataclass(frozen=True, slots=True, repr=False)
class InverseEncoder[X, Y](FittableEncoder[Y, X]):
    r"""Applies an encoder in reverse.

    Example:
        >>> from tsdm.encoders import wrap
        >>> enc = WrappedEncoder(
        ...     encoder=lambda x: f"{x} + 1",
        ...     decoder=lambda y: f"{y} - 1",
        ... )
        >>> assert enc("a") == "a + 1"
        >>> enc = ~enc
        >>> assert enc("a") == "a - 1"
        >>> enc = ~enc
        >>> assert enc("a") == "a + 1"
    """

    FIELDS: ClassVar[frozenset[str]] = frozenset({"encoder"})
    r"""The names of the parameters of the encoder."""

    encoder: Encoder[X, Y]
    r"""The encoder to invert."""

    def fit(self, y: Y, /) -> None:
        raise NotImplementedError("Inverse encoders cannot be fitted.")

    def encode(self, y: Y, /) -> X:
        return self.encoder.decode(y)

    def decode(self, x: X, /) -> Y:
        return self.encoder.encode(x)

    def simplify(self) -> BaseEncoder[Y, X]:
        # reduction 1: simplify the encoder
        e = simplify(self.encoder)

        # reduction 2: remove double inversion
        if isinstance(e, InverseEncoder):
            return simplify(wrap(e.encoder))

        return InverseEncoder(e)

    def __repr__(self) -> str:
        return f"~{self.encoder}"


def invert[X, Y](encoder: Encoder[X, Y], /) -> BaseEncoder[Y, X]:
    r"""Return the inverse encoder (i.e. decoder)."""
    if isinstance(encoder, InverseEncoder):
        # simplify double inversion
        return wrap(encoder.encoder)
    return InverseEncoder(encoder)


# region static encoders ---------------------------------------------------------------
class Diagonal(StaticEncoder[Any, tuple[Any, ...]]):
    r"""Encodes the input into a tuple of itself (SIMO).

             ┌───▶ x
        x ───┼───▶ x
             │     ⋮
             └───▶ x

    Note:
        `Diagonal(n)` is equivalent to `Fork(ID, n)`, where `ID` is the identity encoder.

    Note:
        In practice, when working with float arrays, we need to be careful how to select
        the inverse. Due to rounding errors, the values in the tuple elements might be
        slightly different. In this case, an aggregation function needs to be supplied.
    """

    num: Final[int]
    r"""The number of elements in the tuple."""
    reduction: Final[Reduction[tuple[Any, ...], Any]]
    r"""The function to aggregate the elements of the tuple."""

    def __init__(self, num: int, /, *, reduction: Optional[Reduction] = None) -> None:
        self.num = num
        self.reduction = choice(self.num) if reduction is None else reduction

    def encode[X](self, x: X, /) -> tuple[X, ...]:
        return (x,) * self.num

    def decode[Y](self, y: tuple[Y, ...], /) -> Y:
        return self.reduction(y)

    def simplify(self) -> Self:
        return self


def diagonal[T](num: int, /, reduction: Reduction[tuple[T, ...], T] = random.choice) -> Diagonal:  # fmt: skip
    r"""Encodes the input into a tuple of itself (SIMO).

    Args:
        num: The number of elements in the tuple.
        reduction: The function to aggregate the elements of the tuple.

    Returns:
        A diagonal encoder.
    """
    return Diagonal(num, reduction=reduction)


@pprint_repr
@dataclass
class Choice(StaticEncoder[tuple[Any, ...], Any]):
    r"""Encoder that randomly selects one of the input values.

        x₁ ────┐
        x₂ ────┼────▶ choice(x₁, x₂, ..., xₙ)
        ⋮      │
        xₙ ────┘

    Example:
        >>> from tsdm.encoders import Choice
        >>> enc = Choice(3)
        >>> assert enc((1, 2, 3)) in (1, 2, 3)
        >>> enc = Choice()  # variable number of elements
        >>> assert enc((1, 2)) in (1, 2)
        >>> assert enc((1, 2, 3, 4)) in (1, 2, 3, 4)
    """

    num: Final[int | None] = None
    r"""The number of elements to choose from. If `None`, the encoder will not be invertible."""

    def __invert__(self) -> Diagonal:
        if self.num is None:
            raise ValueError("Choice not invertible when `num=None`.")
        return diagonal(self.num, reduction=self)

    # FIXME: https://discuss.python.org/t/proposal-allow-typevartuple-unpacking-in-unions/
    def encode[X](self, x: tuple[X, ...], /) -> X:
        return random.choice(x)

    # FIXME: https://discuss.python.org/t/proposal-allow-typevartuple-unpacking-in-unions/
    def decode[Y](self, y: Y, /) -> tuple[Y, ...]:
        return (y,) * (self.num or False)

    def simplify(self) -> Self:
        return self


def choice(num: int | None = None, /) -> Choice:
    r"""Create a choice encoder.

    Args:
        num: The number of elements to choose from. If `None`, the encoder will not
            be invertible.

    Returns:
        A choice encoder.
    """
    return Choice(num)


# endregion static encoders ------------------------------------------------------------


# region single input single output encoders -------------------------------------------
@deprecated("Use `Pipe` instead.")
@pprint_repr(recursive=2)
class Compose[X, Y, E: Encoder = Encoder](EncoderList[X, Y, E]):
    r"""Represents function composition of encoders.

    >>> from tsdm.encoders import Compose, wrap
    >>> e1 = wrap(lambda x: f"({x}) + 1")
    >>> e2 = wrap(lambda x: f"2 * ({x})")
    >>> enc = compose(e1, e2)
    >>> assert enc("a") == "(2 * (a)) + 1"
    >>> enc = compose(e2, e1)
    >>> assert enc("a") == "2 * ((a) + 1)"
    """

    @classmethod
    def new[E2: Encoder](cls, *, encoders: Iterable[E2]) -> Compose[Any, Any, E2]:
        return Compose(encoders)

    def __invert__(self) -> Compose[Y, X]:  # type: ignore[override]
        return compose(*map(invert, reversed(self)))

    def fit(self, x: X, /) -> None:
        for encoder in reversed(self):
            try:
                encoder.fit(x)
            except Exception as exc:
                index = self.index(encoder)
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{index}]: Failed to fit {enc!r}.")
                raise
            else:
                x = encoder.encode(x)

    def encode(self, x: X, /) -> Y:
        for encoder in reversed(self):
            x = encoder.encode(x)
        return cast("Y", x)

    def decode(self, y: Y, /) -> X:
        for encoder in self:
            y = encoder.decode(y)
        return cast("X", y)

    def simplify(self) -> BaseEncoder[X, Y]:
        # simplify the nested encoders
        encoders: list[Encoder] = []
        for encoder in map(simplify, self):
            match encoder:
                case Pipe() as pipe:
                    encoders.extend(reversed(pipe))
                case Compose() as chain:
                    encoders.extend(chain)
                case _:
                    encoders.append(encoder)

        # simplify self
        match encoders:
            case []:
                return IdentityEncoder()
            case [encoder]:
                return WrappedEncoder(encoder).simplify()
            case _:
                return Compose(map(simplify, encoders))


# fmt: off
@overload  # n=0
def compose() -> Compose: ...
@overload  # n=1
def compose[X, Y](e: Encoder[X, Y], /) -> Compose[X, Y]: ...
@overload  # n=2
def compose[X, Y, Z](e1: Encoder[Y, Z], e2: Encoder[X, Y], /) -> Compose[X, Z]: ...
@overload  # n>2
def compose[X, Y](*es: *tuple[Encoder[Any, Y], *tuple[Encoder, ...], Encoder[X, Any]]) -> Compose[X, Y]: ...
@overload  # fallback
def compose(*es: Encoder) -> Compose: ...
# fmt: on
@deprecated("Use `pipe` instead.")  # type: ignore[misc]
def compose(*encoders: Encoder) -> Compose:
    r"""Chain encoders.

    See Also: `Compose`, `Pipe`
    """
    return Compose(encoders)


@pprint_repr(recursive=2)
class Pipe[X, Y, E: Encoder = Encoder](EncoderList[X, Y, E]):
    r"""Represents function composition of encoders.

        x ───▶ f₁ ───▶ f₂ ───▶ ... ───▶ fₙ ───▶ y

    Example:
        >>> from tsdm.encoders import Pipe, wrap
        >>> e1 = wrap(lambda x: f"({x}) + 1")
        >>> e2 = wrap(lambda x: f"2 * ({x})")
        >>> enc = e1 >> e2
        >>> assert enc("a") == "2 * ((a) + 1)"
        >>> enc = e2 >> e1
        >>> assert enc("a") == "(2 * (a)) + 1"


    Note that the order is reversed compared to the `@`-operator.

    Note:
        `>>` is associative: `(A >> B) >> C = A >> (B >> C)`

        .. math::
            ((A ≫ B) ≫ C)(x) = C((A ≫ B)(x)) = C(B(A(x)))  \\
            (A ≫ (B ≫ C))(x) = (B ≫ C)(A(x)) = C(B(A(x)))

        .. details:: inverse law: $~(A >> B) == ~B >> ~A$

            .. math::
                &∼(A >> B).encode(x) \\
                    &= (A >> B).decode(x) \\
                    &= B.decode(A.decode(x)) \\
                    &= ∼B.encode(∼~A.encode(x)) \\
                    &= (∼B >> ∼A).encode(x)
    """

    def get_slice(self, arg: slice, /) -> Pipe:
        return Pipe(self.encoders[arg])

    def __invert__(self) -> Pipe[Y, X]:  # type: ignore[override]
        return pipe(*map(invert, reversed(self)))

    def fit(self, x: X, /) -> None:
        for encoder in self:
            try:
                encoder.fit(x)
            except Exception as exc:
                index = self.index(encoder)
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{index}]: Failed to fit {enc!r}.")
                raise
            else:
                x = encoder.encode(x)

    def encode(self, x: X, /) -> Y:
        for encoder in self:
            x = encoder.encode(x)
        return cast("Y", x)

    def decode(self, y: Y, /) -> X:
        for encoder in reversed(self):
            y = encoder.decode(y)
        return cast("X", y)

    def simplify(self) -> BaseEncoder[X, Y]:
        r"""Simplify the chained encoder."""
        # reduction 1: combine nested pipes/chains `e >> (f >> g) = (e >> f) >> g`.
        encoders: list[Encoder] = []
        for encoder in map(simplify, self):
            match encoder:
                case Pipe() as pipe:
                    encoders.extend(pipe)
                case Compose() as chain:
                    encoders.extend(reversed(chain))
                case _:
                    encoders.append(encoder)

        # reduction 2: remove identity encoders `e >> id = e`.
        encoders = [e for e in encoders if not isinstance(e, IdentityEncoder)]

        # reduction 3: remove inverse pairs `(e >> ~e) = id`.

        # reduction 4: remove idempotent encoders `e >> e = e`.

        # reduction 5: combine successive identical encoders via repeat.

        # reduction 6: combine successive identical encoders

        # simplify self
        match encoders:
            case []:
                return IdentityEncoder()
            case [encoder]:
                return WrappedEncoder(encoder).simplify()
            case _:
                return Pipe(map(simplify, encoders))

    @staticmethod
    def _simplify_repeat_encoders(encs: Iterable[Encoder]) -> Iterator[Encoder]:
        r"""Combine successive identical encoders into Repeat."""
        gen = iter(encs)
        succ: None | Encoder = next(gen)  # no try-except necessary

        while succ is not None:
            # get the first encoder and its total count
            match succ:
                case Repeat(encoder=encoder, num=total):
                    pass
                case encoder:
                    total = 1

            succ = None

            for succ in gen:
                if succ is encoder:
                    total += 1
                elif isinstance(succ, Repeat) and succ.encoder is encoder:
                    total += succ.num
                else:
                    # terminate the repeat sequence
                    break

            yield Repeat(encoder, total).simplify()


# fmt: off
@overload  # n=0
def pipe() -> Pipe: ...
@overload  # n=1
def pipe[X, Y](e: Encoder[X, Y], /) -> Pipe[X, Y]: ...
@overload  # n=2
def pipe[X, Y, Z](e1: Encoder[X, Y], e2: Encoder[Y, Z], /) -> Pipe[X, Z]: ...
@overload  # n>2
def pipe[X, Y](*es: *tuple[Encoder[X, Any], *tuple[Encoder, ...], Encoder[Any, Y]]) -> Pipe[X, Y]: ...
@overload  # fallback
def pipe(*es: Encoder) -> Pipe: ...
# fmt: on
def pipe(*es: Encoder) -> Pipe:  # type: ignore[misc]
    r"""Pipe encoders.

    See Also: `Pipe`
    """
    return Pipe(es)


@pprint_repr
@dataclass
class Repeat[T, E: Encoder = Encoder](Pipe[T, T]):
    r"""Repeat copies of an encoder n times (``**``).

        x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)

    Equivalent to `f >> f >> ... >> f` (n times).
    """

    encoder: E
    r"""The encoder to repeat."""
    num: int
    r"""Number of repetitions."""

    def __init__(self, encoder: E, num: int, /) -> None:
        r"""Initialize the encoder."""
        if num < 0:
            raise ValueError("num must be >= 0")
        self.encoder = encoder
        self.num = num
        super().__init__([deepcopy(encoder) for _ in range(num)])

    def get_slice[V](self: Repeat[V], arg: slice, /) -> Repeat[V]:  # fmt: skip
        num = len(self.encoders[arg])
        return Repeat(self.encoder, num)

    def __invert__(self) -> Repeat[T]:
        return repeat(self.encoder, -self.num)

    def simplify(self) -> BaseEncoder[T, T]:
        r"""Simplify the repeat encoder."""
        if isinstance(self.encoder, Repeat):
            # reduce nested repeat encoders
            return Repeat(self.encoder.encoder, self.num * self.encoder.num).simplify()

        if self.num == -1:
            return invert(self.encoder)
        if self.num == 0:
            return IdentityEncoder()
        if self.num == 1:
            return wrap(self.encoder)

        # reduction: combine idempotent encoders (e ** n) = e
        # reduction: combine self-inverse encoder (e >> e) = id
        return Repeat(simplify(self.encoder), self.num)


def repeat[T](e: Encoder[T, T], n: int, /) -> Repeat[T]:
    r"""Repeat an encoder n times (``**``).

        x ───▶ f ──▶ f(x) ──▶ f(f(x)) ──▶ ... ──▶ fⁿ(x)

    See Also: `Repeat`
    """
    return Repeat(e, n)


# endregion single input single output encoders ----------------------------------------


# region multiple input multiple output encoders ---------------------------------------
# FIXME: https://github.com/python/typing/issues/548
#   We could have better type hints with HKTs
# TODO: Use a TypeVarTuple?
@pprint_repr(recursive=2)
class Parallel[
    TupleIn: tuple,  # invariant
    TupleOut: tuple,  # invariant
    E: Encoder = Encoder,  # covariant
](EncoderList[TupleIn, TupleOut, E]):
    r"""Apply multiple encoders in parallel on tuples of data (MIMO).

        x₁ ────▶ f₁(x₁)
        x₂ ────▶ f₂(x₂)
             ⋮
        xₙ ────▶ fₙ(xₙ)

    .. math::
        Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁×…×Xₙ，Y₁×…×Yₙ)  \\
        (f₁，…，fₙ) ⟶ parallel(f₁，…，fₙ)

        parallel(f₁，…，fₙ):
            X₁×…×Xₙ ⟶ Y₁×…×Yₙ   \\
            (x₁，…，xₙ) ⟶ (f₁(x₁)，…，fₙ(xₙ))

    Example:
        >>> from tsdm.encoders import Parallel, wrap
        >>> e1 = wrap(lambda x: f"{x} + 1")
        >>> e2 = wrap(lambda x: f"2 * {x}")
        >>> enc = e1 ^ e2
        >>> assert enc(("a", "b")) == ("a + 1", "2 * b")
    """

    def get_slice[U, V](
        self: Parallel[tuple[U, ...], tuple[V, ...]], arg: slice, /
    ) -> Parallel[tuple[U, ...], tuple[V, ...]]:
        return Parallel(self.encoders[arg])

    def __invert__(self) -> Parallel[TupleOut, TupleIn, Encoder]:
        return cast(
            "Parallel[TupleOut, TupleIn, Encoder]",
            parallel(*map(invert, self)),
        )

    def fit(self, xs: TupleIn, /) -> None:
        for encoder, x in zip(self, xs, strict=True):
            encoder.fit(x)

    def encode(self, xs: TupleIn, /) -> TupleOut:
        return tuple(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
            encoder.encode(x) for encoder, x in zip(self, xs, strict=True)
        )

    def decode(self, ys: TupleOut, /) -> TupleIn:
        return tuple(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
            encoder.decode(x) for encoder, x in zip(self, ys, strict=True)
        )

    def simplify(self) -> BaseEncoder[TupleIn, TupleOut]:
        r"""Simplify the product encoder."""
        # FIXME: https://github.com/python/mypy/issues/17134
        #   Cannot annotate return type as Self!
        match self:
            case []:
                return wrap(  # pyright: ignore[reportReturnType]
                    encoder=lambda _: (),  # type: ignore[arg-type, return-value]
                    decoder=lambda _: (),  # type: ignore[arg-type, return-value]
                )

            case [encoder]:
                return simplify(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    UNWRAP_TUPLE  # [x] -> x
                    >> simplify(encoder)  # x -> y
                    >> WRAP_TUPLE  # y -> [y]
                )

            case _:
                return Parallel(map(simplify, self))


# fmt: off
@overload  # n=0
def parallel() -> Parallel[tuple[()], tuple[()]]: ...
@overload  # n=1
def parallel[X, Y](e: Encoder[X, Y], /) -> Parallel[tuple[X], tuple[Y]]: ...
@overload  # n=2
def parallel[X1, Y1, X2, Y2](e1: Encoder[X1, Y1], e2: Encoder[X2, Y2], /) -> Parallel[tuple[X1, X2], tuple[Y1, Y2]]: ...
@overload  # n>2 (FIXME: https://github.com/python/typing/issues/1216)
def parallel[X, Y](*encoders: Encoder[X, Y]) -> Parallel[tuple[X, ...], tuple[Y, ...]]: ...
@overload  # fallback
def parallel(*encoders: Encoder) -> Parallel[tuple, tuple]: ...
# fmt: on
def parallel(*encoders: Encoder) -> Parallel[tuple, tuple]:
    r"""Apply multiple encoders in parallel on tuples of data (MIMO).

        x₁ ────▶ f₁(x₁)
        x₂ ────▶ f₂(x₂)
             ⋮
        xₙ ────▶ fₙ(xₙ)

    See Also: `Parallel`
    """
    return Parallel(encoders)


@pprint_repr
@dataclass
class Replicate[
    TupleIn: tuple,  # tuple[X, ...]
    TupleOut: tuple,  # tuple[Y, ...]
](Parallel[TupleIn, TupleOut]):
    r"""Apply copies of single encoder in parallel to multiple inputs (MIMO).

        x₁ ────▶ f(x₁)
        x₂ ────▶ f(x₂)
             ⋮
        xₙ ────▶ f(xₙ)

    Example:
        >>> from tsdm.encoders import Replicate, wrap
        >>> e = wrap(lambda x: f"{x} + 1")
        >>> enc = e % 3
        >>> assert enc(("a", "b", "c")) == ("a + 1", "b + 1", "c + 1")
    """

    kind: Final[type[Encoder]]
    num: Final[int]

    # @overload  # n=0
    # def __init__[X, Y](self: "Replicate[tuple[()], tuple[()]]", e: Encoder[X, Y], num: L[0], /) -> None: ...
    # @overload  # n=1
    # def __init__[X, Y](self: "Replicate[tuple[X], tuple[Y]]", e: Encoder[X, Y], num: L[1], /) -> None: ...
    # @overload  # n=2
    # def __init__[X, Y](self: "Replicate[tuple[X, X], tuple[Y, Y]]", e: Encoder[X, Y], num: L[2], /) -> None: ...
    # @overload  # n variable
    # def __init__[X, Y](self: "Replicate[tuple[X, ...], tuple[Y, ...]]", e: Encoder[X, Y], num: int, /) -> None: ...
    def __init__[X, Y](
        self: Replicate[tuple[X, ...], tuple[Y, ...]],
        encoder: Encoder[X, Y],
        num: int,
        /,
    ) -> None:
        if num < 0:
            raise ValueError(f"n must be non-negative, got {num}")
        super().__init__([deepcopy(encoder) for _ in range(num)])
        self.kind = type(self[0]) if self else Encoder
        self.num = len(self)

    @classmethod
    def new[X, Y](  # type: ignore[override]
        cls: type[Replicate], /, *, encoders: Iterable[Encoder[X, Y]]
    ) -> Replicate[tuple[X, ...], tuple[Y, ...]]:
        if cls is not Replicate:
            raise TypeError(f"cls must be Replicate, got {cls}")

        encoders = list(encoders)
        if len({type(e) for e in encoders}) > 1:
            raise TypeError("All encoders must be of the same type.")

        new = Replicate.__new__(Replicate)
        super(Replicate, new).__init__(encoders)
        new.kind = type(encoders[0]) if encoders else Encoder  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]
        new.num = len(encoders)  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]
        return new

    def get_slice[U, V](
        self: Replicate[tuple[U, ...], tuple[V, ...]], arg: slice, /
    ) -> Replicate[tuple[U, ...], tuple[V, ...]]:
        return Replicate.new(encoders=self.encoders[arg])

    def __invert__(self) -> Replicate[TupleOut, TupleIn]:
        # FIXME: https://github.com/python/mypy/issues/20336
        return Replicate.new(encoders=map(invert, self))  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    def simplify(self) -> BaseEncoder[TupleIn, TupleOut]:
        r"""Simplify the replicate encoder."""
        match self:
            case []:
                return wrap(  # pyright: ignore[reportReturnType]
                    encoder=lambda _: (),  # type: ignore[arg-type, return-value]
                    decoder=lambda _: (),  # type: ignore[arg-type, return-value]
                )

            case [encoder]:
                return simplify(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    UNWRAP_TUPLE  # [x] -> x
                    >> simplify(encoder)  # x -> y
                    >> WRAP_TUPLE  # y -> [y]
                )

            case _:
                return Replicate.new(encoders=map(simplify, self))  # type: ignore[return-value]  # pyright: ignore[reportReturnType]


# fmt: off
@overload  # n=0
def replicate[X, Y](e: Encoder[X, Y], num: L[0], /) -> Replicate[tuple[()], tuple[()]]: ...
@overload  # n=1
def replicate[X, Y](e: Encoder[X, Y], num: L[1], /) -> Replicate[tuple[X], tuple[Y]]: ...
@overload  # n=2
def replicate[X, Y](e: Encoder[X, Y], num: L[2], /) -> Replicate[tuple[X, X], tuple[Y, Y]]: ...
@overload  # n variable
def replicate[X, Y](e: Encoder[X, Y], num: int, /) -> Replicate[tuple[X, ...], tuple[Y, ...]]: ...
# fmt: on
def replicate[X, Y](e: Encoder[X, Y], num: int, /) -> Replicate[tuple[X, ...], tuple[Y, ...]]:  # type: ignore[misc]  # fmt: skip
    r"""Create copies of an Encoder in parallel.

        x₁ ────▶ f(x₁)
        x₂ ────▶ f(x₂)
             ⋮
        xₙ ────▶ f(xₙ)

    Args:
        e: The encoder to duplicate.
        num: The number of copies. Must be non-negative.
    """
    return Replicate(e, num)


# endregion multiple input multiple output encoders ------------------------------------


# region single input multiple output encoders -----------------------------------------
# @dataclass
# class Expand[X, TupleOut: tuple ](FittableEncoder[X, TupleOut]):
#     r"""Encoder that expands the input into a tuple of values (Single Input Multiple Outputs).
#
#               ┌────▶ y₁
#         x ────┼────▶ y₂
#               │       ⋮
#               └────▶ yₙ
#
#     See Also: `expand`
#     """
#
#     expansion: Encoder
#     reduction: Fn[[TupleOut], X]
#     r"""The inverse, if applicable."""
#
#     def fit(self, x: X, /) -> None:
#         self.expansion.fit(x)
#
#     def encode(self, x: X, /) -> TupleOut:
#         return self.expansion.encode(x)
#
#     def decode(self, ys: TupleOut, /) -> X:
#         return self.reduction(ys)
#
#
# def expand[X, TupleOut: tuple ](
#     expansion: Encoder[X, TupleOut],
#     reduction: Fn[[TupleOut], X] = lambda ys: ys[0],
# ) -> Expand[X, TupleOut]:
#     r"""Create an encoder that expands the input into a tuple of values.
#
#     Args:
#         expansion: The encoder to expand the input.
#         reduction: The function to reduce the tuple back to the original input.
#
#     Returns:
#         An encoder that expands the input into a tuple of values.
#     """
#     return Expand(expansion=expansion, reduction=reduction)


# TODO: Use TypeVarTuple?
@pprint_repr(recursive=2)
class Fork[
    X,  # invariant
    TupleOut: tuple,  # invariant
    E: Encoder = Encoder[X, Any],  # covariant
](EncoderList[X, TupleOut, E]):
    r"""Apply multiple encoders to the same input (SIMO).

              ┌────▶ f₁(x)
        x ────┼────▶ f₂(x)
              │        ⋮
              └────▶ fₙ(x)

    .. math::
        Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁∩…∩Xₙ，Y₁×…×Yₙ)  \\
        (f₁，…，fₙ) ⟼ fork(f₁，…，fₙ)

        fork(f₁，…，fₙ):
            X₁∩…∩Xₙ ⟶ Y₁×…×Yₙ   \\
            x ⟼ (f₁(x), ..., fₙ(x))

    Note:
        `Fork` is essentially a heterogeneous `Split` encoder.
        `Fork(f1, ..., fn)` is equivalent to `Diagonal(n) >> Parallel(f1, ..., fn)`.
        Hence, `~Fork(f1, ..., fn) = `Parallel(~f1, ..., ~fn) >> Choice(n)`.

    Example:
        >>> from tsdm.encoders import Fork, wrap
        >>> e1 = wrap(lambda x: f"{x} + 1")
        >>> e2 = wrap(lambda x: f"2 * {x}")
        >>> e3 = wrap(lambda x: f"{x}**3")
        >>> enc = Fork(e1, e2, e3)
        >>> assert enc("a") == ("a + 1", "2 * a", "a**3")
        >>> enc = e1 | e2
        >>> assert enc("a") == ("a + 1", "2 * a")
    """

    # NOTE: Need to use different variable names than the class-scoped parameters!
    # fmt: off
    # @overload  # n=0
    # def __init__[T](self: "Fork[T, tuple[()]]", *, reduction: Reduction[tuple[()], T] = ...) -> None: ...
    # @overload  # n=1
    # def __init__[T, Y=Any](self: "Fork[T, tuple[Y]]", e: Encoder[T, Y], /, *, reduction: Reduction[tuple[T], T] = ...) -> None: ...
    # @overload  # n=2
    # def __init__[T, Y1, Y2](self: "Fork[T, tuple[Y1, Y2]]", e1: Encoder[T, Y1], e2: Encoder[T, Y2], /, *, reduction: Reduction[tuple[T, T], T] = ...) -> None: ...
    # @overload  # n>2 same output type
    # def __init__[T, Y=Any](self: "Fork[T, tuple[Y, ...]]", *es: Encoder[T, Y], reduction: Reduction[tuple[T, ...], T] = ...) -> None: ...
    # @overload  # n>2 different output types
    # def __init__[T, Vs: tuple](self: "Fork[T, Vs]", *encoders: Encoder[T, Any], reduction: Reduction[Vs, T] = ...) -> None: ...
    # fmt: on
    def __init__(
        self,
        *encoders: Encoder[X, Any],  # *(Encoder[X, Y] for Y in Ys),
        reduction: Reduction[tuple[X, ...], X] = random.choice,
    ) -> None:
        super().__init__(encoders)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        # self.expansion = diagonal(len(encoders)) >> parallel(*encoders)
        self.reduction: Final[Reduction[tuple[X, ...], X]] = reduction

    # FIXME: possibly incorrect for inhomogeneous reductions
    def get_slice[U, V](self: Fork[U, tuple[V, ...]], arg: slice, /) -> Fork[U, tuple[V, ...]]:  # fmt: skip
        return Fork(*self.encoders[arg], reduction=self.reduction)

    def __invert__(self) -> Meet[TupleOut, X]:
        return cast(
            "Meet[TupleOut, X]",
            meet(*map(invert, self), reduction=self.reduction),
        )

    def fit(self, x: X, /) -> None:
        r"""Fit all encoders in the fork."""
        for encoder in self:
            encoder.fit(x)

    def encode(self, x: X, /) -> TupleOut:
        return tuple(e.encode(x) for e in self)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    def decode(self, ys: TupleOut, /) -> X:
        decoded_vals = tuple(e.decode(y) for e, y in zip(self, ys, strict=True))
        return self.reduction(decoded_vals)

    def simplify(self) -> BaseEncoder[X, TupleOut]:
        match self:
            case []:  # encode[any X -> ()], decode[() -> some x] (depends on reduction)
                return simplify(  # pyright: ignore[reportReturnType]
                    wrap(
                        encoder=lambda _: (),  # type: ignore[return-value]
                        decoder=self.reduction,
                    )
                )

            case [encoder]:  # encode[X -> [f(x)]], decode[[y] -> f⁻¹(reduce([y]))]
                return simplify(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    cast("Encoder[X, Any]", encoder)
                    >> wrap(
                        encoder=WRAP_TUPLE,
                        decoder=self.reduction,
                    )
                )

            case _:
                return Fork(*map(simplify, self))


# fmt: off
@overload  # n=0
def fork[X=Any](*, reduction: Reduction[tuple[()], X] = ...) -> Fork[X, tuple[()]]: ...  # pyright: ignore[reportInvalidTypeVarUse]
@overload  # n=1
def fork[X, Y](e: Encoder[X, Y], /, *, reduction: Reduction[tuple[X, X], X] = ...) -> Fork[X, tuple[Y]]: ...
@overload  # n=2
def fork[X, Y1, Y2](e1: Encoder[X, Y1], e2: Encoder[X, Y2], /, *, reduction: Reduction[tuple[X, X], X] = ...) -> Fork[X, tuple[Y1, Y2]]: ...
@overload  # n>2
def fork[X, Y](*es: Encoder[X, Y], reduction: Reduction[tuple[X, ...], X] = ...) -> Fork[X, tuple[Y, ...]]: ...
# fmt: on
def fork[X, Y](  # type: ignore[misc]  # pyright: ignore[reportInconsistentOverload]
    *encoders: Encoder[X, Y],
    reduction: Reduction[tuple[X, ...], X] = random.choice,
) -> Fork[X, tuple[Y, ...]]:
    r"""Apply multiple encoders to the same input (SIMO).

              ┌────▶ f₁(x)
        x ────┼────▶ f₂(x)
              │        ⋮
              └────▶ fₙ(x)

    See Also:  `Fork`
    """
    return Fork(*encoders, reduction=reduction)


@pprint_repr
@dataclass
class Duplicate[
    X,
    Ys: tuple,  # tuple[Y, Y, ..., Y]
    # FIXME: https://github.com/python/cpython/issues/140596
    E: Encoder = Encoder[X, Any],  # Encoder[X, Y]
](Fork[X, Ys, E]):
    r"""Apply copies of a single encoder to the same input (SIMO).

              ┌────▶ f(x)
        x ────┼────▶ f(x)
              │        ⋮
              └────▶ f(x)

    .. math::
        Fun(X，Y) × ℕ ⟶ Fun(X，Y×…×Y)
        (f，n) ⟼ duplicate(f, n)

        duplicate(f, n):
            X ⟶ Y×…×Y
            x ⟼ (f(x), ..., f(x))

    Note:
        `Split` is essentially a homogeneous `Fork` encoder.
        `Split(f, n)` is equivalent to `Diagonal(n) >> Replicate(f, n)`.
        Hence, `~Split(f, n) = Duplicate(~f, n) >> Choice(n) = Fold(~f, n)`.

    Example:
        >>> from tsdm.encoders import Duplicate, wrap
        >>> e = wrap(lambda x: f"{x} + 1")
        >>> enc = e * 3
        >>> assert enc("a") == ("a + 1", "a + 1", "a + 1")
    """

    num: Final[int]
    reduction: Final[Reduction[tuple[X, ...], X]]  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]

    # fmt: off
    # @overload  # n=0
    # def __init__[T, Y](self: "Duplicate[T, tuple[()]]", e: Encoder[T, Y], num: L[0], /, *, reduction: Reduction[tuple[()], T] = ...) -> None: ...
    # @overload  # n=1
    # def __init__[T, Y](self: "Duplicate[T, tuple[Y]]", e:  Encoder[T, Y], num: L[1], /, *, reduction: Reduction[tuple[T], T] = ...) -> None: ...
    # @overload  # n=2
    # def __init__[T, Y](self: "Duplicate[T, tuple[Y, Y]]", e:  Encoder[T, Y], num: L[2], /, *, reduction: Reduction[tuple[T, T], T] = ...) -> None: ...
    # @overload  # n>2
    # def __init__[T, Y](self: "Duplicate[T, tuple[Y, ...]]", e:  Encoder[T, Y], num: int, /, *, reduction: Reduction[tuple[T, ...], T] = ...) -> None: ...
    # fmt: on
    def __init__[U, V](
        self: Duplicate[U, tuple[V, ...]],
        encoder: Encoder[U, V],  # Encoder[X, Y]
        num: int,
        /,
        *,
        reduction: Reduction[tuple[U, ...], U] = random.choice,
    ) -> None:
        super().__init__(*(deepcopy(encoder) for _ in range(num)), reduction=reduction)
        self.num = num

    @classmethod
    def new[Y](  # type: ignore[override]
        cls: type[Duplicate],
        /,
        *,
        encoders: Iterable[Encoder[X, Y]],
        reduction: Reduction[tuple[X, ...], X] = random.choice,
    ) -> Duplicate[X, tuple[Y, ...]]:
        if cls is not Duplicate:
            raise TypeError(f"cls must be Duplicate, got {cls}")

        encoders = list(encoders)
        if len({type(e) for e in encoders}) > 1:
            raise TypeError("All encoders must be of the same type.")

        new = Duplicate.__new__(Duplicate)
        super(Duplicate, new).__init__(*encoders, reduction=reduction)
        new.num = len(encoders)  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]
        return new

    def get_slice[U, V](self: Duplicate[U, tuple[V, ...]], arg: slice, /) -> Duplicate[U, tuple[V, ...]]:  # fmt: skip
        return Duplicate.new(encoders=self.encoders[arg], reduction=self.reduction)

    def __invert__(self) -> Fold[Ys, X]:
        return Fold.new(encoders=map(invert, self), reduction=self.reduction)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    def simplify(self) -> BaseEncoder[X, Ys]:
        r"""Simplify the duplicate encoder."""
        match self:
            case []:  # encode[any x -> ()], decode[() -> some x] (depends on reduction)
                return wrap(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    encoder=lambda _: (),
                    decoder=self.reduction,
                ).simplify()

            case [encoder]:  # encode[X -> [f(x)]], decode[[y] -> f⁻¹(reduce([y]))]
                return (  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    cast("Encoder[X, Any]", encoder)
                    >> wrap(
                        encoder=WRAP_TUPLE,
                        decoder=self.reduction,
                    )
                ).simplify()

            case _:
                return Duplicate.new(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    encoders=map(simplify, self),
                    reduction=self.reduction,
                )


# fmt: off
@overload  # n=0
def duplicate[X, Y](
    e: Encoder[X, Y], num: L[0], /, *, reduction: Reduction[tuple[()], X] = ...
) -> Fork[X, tuple[()]]: ...
@overload  # n=1
def duplicate[X, Y](
    e: Encoder[X, Y], num: L[1], /, *, reduction: Reduction[tuple[X], X] = ...
) -> Fork[X, tuple[Y]]: ...
@overload  # n=2
def duplicate[X, Y](
    e: Encoder[X, Y], num: L[2], /, *, reduction: Reduction[tuple[X, X], X] = ...
) -> Fork[X, tuple[Y, Y]]: ...
@overload  # n>2
def duplicate[X, Y](
    e: Encoder[X, Y], num: int, /, *, reduction: Reduction[tuple[X, ...], X] = ...
) -> Fork[X, tuple[Y, ...]]: ...
# fmt: on
def duplicate[X, Y](  # type: ignore[misc]
    e: Encoder[X, Y], num: int, /, *, reduction: Reduction[tuple, X] = random.choice
) -> Fork[X, tuple[Y, ...]]:
    r"""Apply copies of a single encoder to the same input (SIMO).

              ┌────▶ f(x)
        x ────┼────▶ f(x)
              │        ⋮
              └────▶ f(x)

    Args:
        e: The encoder to fork.
        num: The number of forks. Must be non-negative.
        reduction: The function to reduce the tuple back to the original input.

    Returns:
        A `Fork` encoder that applies the same encoder to the input multiple times.
    """
    return Duplicate(e, num, reduction=reduction)


# endregion single input multiple output encoders --------------------------------------


# region single input multiple output encoders -----------------------------------------
# @pprint_repr
# @dataclass
# class Reduce[T, E: Encoder](FittableEncoder[tuple[T, ...], T]):
#     r"""Encoder that reduces the input to a single value.
#
#         x₁ ────┐
#         x₂ ────┼────▶ aggregate(x₁, x₂, ..., xₙ)
#         ⋮      │
#         xₙ ────┘
#
#     Examples:
#         - `random.choice` for generic data
#         - `min`, `max`, `median` for ordered data
#         - `any`, `all` for boolean data
#         - `sum`, `mean`, `prod`, `std`, `var`, `logsumexp` for numerical data
#         - `stack`, `concat` for tensor data
#     """
#
#     reduction: Reduction[tuple[T, ...], T] | None = None
#     expansion: Expansion[T, tuple[T, ...]] | None = None
#
#     def __invert__(self) -> "BaseEncoder[T, tuple[T, ...]]":
#         raise NotImplementedError
#
#     def fit(self, x: tuple[T, ...], /) -> None:
#         pass
#
#     def encode(self, x: tuple[T, ...], /) -> T:
#         return self.reduction(x)
#
#     def decode(self, y: T, /) -> tuple[T, ...]:
#         return self.expansion(y)
#
#
# def reduce[T, E: Encoder](
#     reduction: Reduction[tuple[T, ...], T] | None = None,
#     expansion: Expansion[T, tuple[T, ...]] | None = None,
# ) -> Reduce[T, E]:
#     r"""Create a reduction encoder.
#
#     Args:
#         reduction: The function to reduce the input to a single value.
#         expansion: The function to expand the single value back to a tuple of values.
#
#     Returns:
#         A reduction encoder.
#     """
#     return Reduce(reduction=reduction, expansion=expansion)


@pprint_repr(recursive=2)
class Meet[TupleIn: tuple, Y, E: Encoder = Encoder](EncoderList[TupleIn, Y, E]):
    r"""Combine the outputs of multiple encoders into a single value (MISO).

        x₁ ────┐
        x₂ ────┼────▶ reduction([f₁(x₁), f₂(x₂), ..., fₙ(xₙ)])
        ⋮      │
        xₙ ────┘

    .. math::
            Fun(X₁，Y₁) × … × Fun(Xₙ，Yₙ) ⟶ Fun(X₁×…×Xₙ，Y₁∪…∪Yₙ)  \\
            (f₁，…，fₙ) ⟼ join(f₁，…，fₙ)

        join(f₁，…，fₙ):
            X₁×…×Xₙ ⟶ Y₁∪…∪Yₙ \\
            x ⟼ reduction(f₁(x), …, fₙ(x))


    Assumptions:
        - All encoders map into the same type `Y`.
        - The reduction function combines the outputs of the encoders into a single value.


    Example:
        >>> from tsdm.encoders import Meet, wrap
        >>> e = wrap(lambda x: f"({x} + 1)")
        >>> enc = Meet(e, e, e, reduction="*".join)
        >>> assert enc(("a", "b", "c")) == "(a + 1)*(b + 1)*(c + 1)"

    Examples:
        - `random.choice` for generic data
        - `min`, `max`, `median` for ordered data
        - `any`, `all` for boolean data
        - `sum`, `mean`, `prod`, `std`, `var`, `logsumexp` for numerical data
        - `stack`, `concat` for tensor data
    """

    # NOTE: Need to use different variable names than the class-scoped parameters!
    # fmt: off
    # @overload  # n=0
    # def __init__[Z](self: "Reduce[tuple[()], Z]", *, aggregate_fn: Agg[Z] = ...) -> None: ...
    # @overload  # n=1
    # def __init__[X, Z](self: "Reduce[tuple[X], Z]", e: Encoder[X, Z], /, *, aggregate_fn: Agg[Z] = ...) -> None: ...
    # @overload  # n=2
    # def __init__[X1, X2, Z](self: "Reduce[tuple[X1, X2], Z]", e1: Encoder[X1, Z], e2: Encoder[X2, Z], /, *, aggregate_fn: Agg[Z] = ...) -> None: ...
    # @overload  # n>2
    # def __init__[X, Z](self: "Reduce[tuple[X, ...], Z]", *es: Encoder[X, Z], aggregate_fn: Agg[Z] = ...) -> None: ...
    # fmt: on
    def __init__(
        self,
        *encoders: Encoder[Any, Y],  # *(Encoder[X, Y] for X in Xs)
        reduction: Reduction[tuple[Y, ...], Y] = random.choice,
    ) -> None:
        super().__init__(encoders)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        self.reduction: Final[Reduction[tuple[Y, ...], Y]] = reduction

    # FIXME: possibly incorrect for inhomogeneous reductions
    def get_slice[U, V](
        self: Meet[tuple[U, ...], V], arg: slice, /
    ) -> Meet[tuple[U, ...], V]:
        return Meet(*self.encoders[arg], reduction=self.reduction)

    def __invert__(self) -> Fork[Y, TupleIn]:
        return cast(
            "Fork[Y, TupleIn]",
            fork(*map(invert, self), reduction=self.reduction),
        )

    def fit(self, xs: TupleIn, /) -> None:
        for x, e in zip(xs, self, strict=True):
            e.fit(x)

    def encode(self, xs: TupleIn, /) -> Y:
        encoded_vals = tuple(e.encode(x) for e, x in zip(self, xs, strict=True))
        return self.reduction(encoded_vals)

    def decode(self, y: Y, /) -> TupleIn:
        return tuple(e.decode(y) for e in self)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    def simplify(self) -> BaseEncoder[TupleIn, Y]:
        r"""Simplify the joint encoder."""
        if type(self) is not Meet:
            raise TypeError("Subclasses must override `simplify`.")

        match self:
            case []:  # encode[() -> some x], decode[any x -> ()]  (depends on reduction)
                return wrap(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    encoder=self.reduction,
                    decoder=lambda _: (),
                ).simplify()

            case [encoder]:  # encode[[x] -> f(reduce([x])], decode[y -> [f⁻¹y]))]
                return simplify(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    wrap(
                        encoder=self.reduction,
                        decoder=WRAP_TUPLE,
                    )
                    >> cast("Encoder[Any, Y]", encoder)
                )

            case _:
                return Meet(*map(simplify, self), reduction=self.reduction)


# fmt: off
@overload  # n=0
def meet[Y = Any](
    *, reduction: Reduction[tuple[()], Y] = ...  # pyright: ignore[reportInvalidTypeVarUse]
) -> Meet[tuple[()], Y]: ...
@overload  # n=1
def meet[X, Y](
    e: Encoder[X, Y], /, *, reduction: Reduction[tuple[Y], Y] = ...
) -> Meet[tuple[X], Y]: ...
@overload  # n=2
def meet[X, X2, Y](
    e1: Encoder[X, Y], e2: Encoder[X2, Y], /,  *, reduction: Reduction[tuple[Y, Y], Y] = ...,
) -> Meet[tuple[X, X2], Y]: ...
@overload  # n>2
def meet[X, Y](
    *es: Encoder[X, Y], reduction: Reduction[tuple[Y, ...], Y] = ...
) -> Meet[tuple[X, ...], Y]: ...
# fmt: on
def meet[Y](
    *es: Encoder[Any, Y], reduction: Reduction[tuple, Y] = random.choice
) -> Meet[tuple, Any]:
    r"""Combine the outputs of multiple encoders into a single value (MISO).

        x₁ ────┐
        x₂ ────┼────▶ reduction([f₁(x₁), f₂(x₂), ..., fₙ(xₙ)])
        ⋮     │
        xₙ ────┘

    See Also: `Join`
    """
    return Meet(*es, reduction=reduction)


class Fold[Xs: tuple, Y](Meet[Xs, Y]):  # (tuple[X, ...], Y]):
    r"""Apply copies of a single encoder to multiple inputs and reduces (MISO).

        x₁ ────┐
        x₂ ────┼────▶ reduction([f(x₁), f(x₂), ..., f(xₙ)])
        ⋮      │
        xₙ ────┘

    .. math::
        Fun(X，Y) × ℕ ⟶ Fun(X×…×X，Y)
        (f, n) ⟼ fold(f, n)

        fold(f, n):
            X×…×X ⟶ Y
            x ⟼ reduction(f(x), …, f(x))

    Note:
        `Fold` is essentially a homogeneous `Join` encoder.
        `Fold(f, n)` is equivalent to `Duplicate(f, n) >> Choice(n)`.
        Hence, `~Fold(f, n) = Diagonal(n) >> Duplicate(~f, n) = Split(~f, n)`.

    Example:
        >>> from tsdm.encoders import Fold, wrap
        >>> e = wrap(lambda x: f"({x} + 1)")
        >>> enc = Fold(e, 3, reduction="*".join)
        >>> assert enc(("a", "b", "c")) == "(a + 1)*(b + 1)*(c + 1)"
    """

    kind: Final[type[Encoder[Any, Y]]]  # type: ignore[misc]
    num: Final[int]
    reduction: Final[Reduction[tuple[Y, ...], Y]]  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]

    def __init__[U, V](
        self: Fold[tuple[U, ...], V],
        encoder: Encoder[U, V],
        num: int,
        /,
        *,
        reduction: Reduction[tuple[V, ...], V] = random.choice,
    ) -> None:
        super().__init__(*(deepcopy(encoder) for _ in range(num)), reduction=reduction)
        self.kind = type(self[0]) if self else Encoder
        self.num = num

    @classmethod
    def new[X](  # type: ignore[override]
        cls: type[Fold],
        /,
        *,
        encoders: Iterable[Encoder[X, Y]],
        reduction: Reduction[tuple[Y, ...], Y] = random.choice,
    ) -> Fold[tuple[X, ...], Y]:
        if cls is not Fold:
            raise TypeError(f"cls must be Fold, got {cls}")

        encoders = list(encoders)
        if len({type(e) for e in encoders}) > 1:
            raise TypeError("All encoders must be of the same type.")

        new = Fold.__new__(Fold)
        super(Fold, new).__init__(*encoders, reduction=reduction)
        new.kind = type(encoders[0]) if encoders else Encoder  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]
        new.num = len(encoders)  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]
        return new

    def get_slice[U, V](self: Fold[tuple[U, ...], V], arg: slice, /) -> Fold[tuple[U, ...], V]:  # fmt: skip
        return Fold.new(encoders=self.encoders[arg], reduction=self.reduction)

    def __invert__(self) -> Duplicate[Y, Xs]:
        return cast(
            "Duplicate[Y, Xs]",
            Duplicate.new(encoders=map(invert, self), reduction=self.reduction),
        )

    def simplify(self) -> BaseEncoder[Xs, Y]:
        r"""Simplify the fold encoder."""
        match self:
            case []:
                return wrap(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    encoder=self.reduction,
                    decoder=lambda _: (),
                ).simplify()

            case [encoder]:
                encoder = cast("Encoder[Any, Y]", encoder)
                return (  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    wrap(
                        encoder=self.reduction,
                        decoder=WRAP_TUPLE,
                    )
                    >> encoder
                ).simplify()

            case _:
                return Fold.new(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
                    encoders=map(simplify, self),
                    reduction=self.reduction,
                )


@overload  # n=0
def fold[X, Y](
    e: Encoder[X, Y], num: L[0], /, *, reduction: Reduction[tuple[Y, ...], Y] = ...
) -> Meet[tuple[()], Y]: ...
@overload  # n=1
def fold[X, Y](
    e: Encoder[X, Y], num: L[1], /, *, reduction: Reduction[tuple[Y], Y] = ...
) -> Meet[tuple[X], Y]: ...
@overload  # n=2
def fold[X, Y](
    e: Encoder[X, Y], num: L[2], /, *, reduction: Reduction[tuple[Y, Y], Y] = ...
) -> Meet[tuple[X, X], Y]: ...
@overload  # n>2
def fold[X, Y](
    e: Encoder[X, Y], num: int, /, *, reduction: Reduction[tuple[Y, ...], Y] = ...
) -> Meet[tuple[X, ...], Y]: ...
def fold[X, Y](  # type: ignore[misc]
    e: Encoder[X, Y], num: int, /, *, reduction: Reduction[tuple, Y] = random.choice
) -> Meet[tuple[X, ...], Y]:
    r"""Apply copies of a single encoder to multiple inputs and reduces (MISO).

        x₁ ────┐
        x₂ ────┼────▶ reduction([f(x₁), f(x₂), ..., f(xₙ)])
        ⋮      │
        xₙ ────┘

    Args:
        e: The encoder to fork.
        num: The number of forks. Must be non-negative.
        reduction: Function to reduce the tuple of outputs to a single value.

    Returns:
        An encoder that applies the same encoder to the input multiple times and reduces the outputs.
    """
    return Fold(e, num, reduction=reduction)


# endregion single input multiple output encoders --------------------------------------
# endregion ALGEBRAIC INTERFACE ########################################################
