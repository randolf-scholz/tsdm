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
"""
# ruff: noqa: E501

__all__ = [
    # ABCs & Protocols
    "BackendMixin",
    "BaseEncoder",
    "Encoder",
    "EncoderDict",
    "EncoderList",
    "EncoderProtocol",
    "InvertibleTransform",
    "ParametrizedEncoder",
    "SerializableEncoder",
    "Transform",
    "UniversalEncoder",
    # Classes
    "ChainedEncoder",
    "DeepcopyEncoder",
    "DiagonalDecoder",
    "DiagonalEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "JointDecoder",
    "JointEncoder",
    "MappedEncoder",
    "NestedEncoder",
    "ParallelEncoder",
    "PipedEncoder",
    "TupleDecoder",
    "TupleEncoder",
    "WrappedEncoder",
    # Functions
    "chain_encoders",
    "duplicate_encoder",
    "invert_encoder",
    "join_encoders",
    "map_encoders",
    "parallelize_encoders",
    "pipe_encoders",
    "pow_encoder",
]

import logging
import pickle
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import KW_ONLY, asdict, dataclass
from functools import wraps
from inspect import getattr_static
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    cast,
    overload,
    runtime_checkable,
)

from tsdm import encoders as E
from tsdm.backend import Backend, get_backend
from tsdm.constants import EMPTY_MAP
from tsdm.types.aliases import FilePath, NestedBuiltin
from tsdm.types.protocols import Dataclass, SupportsKeysAndGetItem
from tsdm.utils.decorators import pprint_mapping, pprint_repr, pprint_sequence

type Agg[T] = Callable[[list[T]], T]


# region protocol classes --------------------------------------------------------------
@runtime_checkable
class Transform[X, Y](Protocol):  # -X, +Y
    r"""Protocol for transformers."""

    @abstractmethod
    def fit(self, x: X, /) -> None: ...
    @abstractmethod
    def transform(self, x: X, /) -> Y: ...


@runtime_checkable
class InvertibleTransform[X, Y](Transform[X, Y], Protocol):
    r"""Protocol for invertible transformers."""

    @abstractmethod
    def inverse_transform(self, y: Y, /) -> X: ...


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
        return any(params[key] is NotImplemented for key in self.required_params)

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        r"""Alias for `self.params`."""
        return self.params

    # endregion mixin methods ---------------------------------------------------------


class SerializableEncoder[X, Y](EncoderProtocol[X, Y], Protocol):
    r"""Protocol for serializable encoders."""

    @classmethod
    @abstractmethod
    def deserialize(cls, filepath: FilePath, /) -> Self: ...
    @abstractmethod
    def serialize(self, filepath: FilePath, /) -> None: ...


class Encoder[X, Y](Protocol):
    r"""Protocol for Encoders with algebraic mixin methods."""

    # region abstract methods ----------------------------------------------------------
    @property
    @abstractmethod
    def is_fitted(self, /) -> bool: ...
    @is_fitted.setter
    @abstractmethod
    def is_fitted(self, value: bool, /) -> None: ...
    @property
    @abstractmethod
    def params(self) -> dict[str, Any]: ...
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

    @property
    def required_params(self) -> frozenset[str]:
        r"""The required parameters to initialize the encoder."""
        return frozenset(self.params)

    def simplify(self) -> "Encoder[X, Y]":
        r"""Simplify the encoder."""
        return self

    # region serialization methods -----------------------------------------------------
    def is_serializable(self) -> bool:
        r"""Check if the encoder is serializable."""
        params = self.params
        return not any(params[key] is NotImplemented for key in self.required_params)

    def serialize(self, filepath: FilePath, /) -> None:
        r"""Serialize the encoder to a file."""
        if not self.is_serializable():
            raise RuntimeError("Encoder is not serializable!")

        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def deserialize(cls, filepath: FilePath, /) -> Self:
        r"""Deserialize the encoder from a file."""
        with open(filepath, "rb") as file:
            obj = pickle.load(file)
            if not isinstance(obj, cls):
                raise TypeError(f"Deserialized object is not an instance of {cls}.")
        return obj

    # endregion serialization methods --------------------------------------------------

    # region scikit-learn compatibility ------------------------------------------------
    def fit_transform(self, x: X, /) -> Y:
        r"""Fit the encoder to the data and apply the transformation."""
        self.fit(x)
        return self.encode(x)

    def transform(self, x: X, /) -> Y:
        r"""Alias for encode."""
        return self.encode(x)

    def inverse_transform(self, y: Y, /) -> X:
        r"""Alias for decode."""
        return self.decode(y)

    def set_params(self, **kwargs: Any) -> None:
        r"""Compatibility interface with scikit-learn."""
        try:
            self.__dict__.update(**kwargs)
        except Exception as exc:
            exc.add_note(f"failed to set parameters {kwargs}.")
            raise

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Compatibility interface with scikit-learn.

        Compatibility interface with scikit-learn.
        Use the property `self.params` to access the parameters directly.

        Uninitialized required parameters should be marked as `NotImplemented`.
        Uninitialized optional parameters should be marked as `None`.
        """
        return self.params

    # endregion scikit-learn compatibility ---------------------------------------------

    # region magic methods -------------------------------------------------------------
    # NOTE: We exclude the magic methods from the protocol, because diverging
    #   protocols raise recursion error in mypy. They might be added later.
    # FIXME: https://github.com/python/mypy/issues/17326
    # endregion magic methods ----------------------------------------------------------


# endregion protocol classes -----------------------------------------------------------


# region base classes ------------------------------------------------------------------
class BaseEncoder[X, Y](Encoder[X, Y]):
    r"""Base class for encoders implemented within this package."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the Encoder."""

    def __init_subclass__(cls) -> None:
        r"""Initialize the subclass.

        The wrapping of fit/encode/decode must be done here to avoid `~pickle.PickleError`!
        """
        super().__init_subclass__()  # <-- This is important! Otherwise, weird things happen.

        # validate that the methods are implemented
        for meth in ("fit", "encode", "decode"):
            static_meth = getattr_static(cls, meth, None)
            if static_meth is None:
                raise NotImplementedError(f"Missing method {meth}.")
            if isinstance(static_meth, staticmethod | classmethod):
                raise TypeError(f"Method {meth} can't be static/class method.")

        # wrap the methods
        original_fit = cls.fit
        original_encode = cls.encode
        original_decode = cls.decode

        @wraps(original_fit)
        def fit_wrapper(self: Self, x: X, /) -> None:
            r"""Fit the encoder to the data."""
            self.LOGGER.info("Fitting encoder to data.")
            original_fit(self, x)

            # check if fitting was successful
            if self.requires_fit:
                msg = "Fitting was not successful, the encoder still requires fitting!"

                if bad_params := {
                    key: val
                    for key, val in self.params.items()
                    if val is NotImplemented or getattr(val, "requires_fit", False)
                }:
                    msg += (
                        f"\nThis is likely because the following parameters are not set correctly:"
                        f"\n{bad_params}"
                    )

                raise AssertionError(msg)
            self.is_fitted = True

        @wraps(original_encode)
        def encode_wrapper(self: Self, x: X, /) -> Y:
            r"""Encode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted!")
            return original_encode(self, x)

        @wraps(original_decode)
        def decode_wrapper(self: Self, y: Y, /) -> X:
            r"""Decode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted!")
            return original_decode(self, y)

        cls.fit = fit_wrapper  # type: ignore[assignment]
        cls.encode = encode_wrapper  # type: ignore[assignment]
        cls.decode = decode_wrapper  # type: ignore[assignment]

    _is_fitted: bool = False
    r"""Whether the encoder has been fitted."""

    @property
    def params(self) -> dict[str, Any]:
        if isinstance(self, Dataclass):
            return asdict(self)
        raise NotImplementedError("Method `params` must be implemented.")

    @property
    def requires_fit(self) -> bool:
        r"""Check if the encoder requires fitting."""
        return any(
            val is NotImplemented or getattr(val, "requires_fit", False)
            for val in self.params.values()
        )

    @property
    def is_fitted(self) -> bool:
        r"""Whether the encoder has been fitted."""
        return (not self.requires_fit) or self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool, /) -> None:
        self._is_fitted = value

    # region abstract methods ----------------------------------------------------------

    @abstractmethod
    def encode(self, x: X, /) -> Y:
        r"""Encode the data by transformation."""
        ...

    @abstractmethod
    def decode(self, y: Y, /) -> X:
        r"""Decode the data by inverse transformation."""
        ...

    def fit(self, x: X, /) -> None:
        r"""Implement as necessary."""

    def simplify(self) -> "Encoder[X, Y]":
        r"""Simplify the encoder."""
        return self

    # endregion abstract methods -------------------------------------------------------

    # region magic methods -------------------------------------------------------------
    def __invert__(self) -> "BaseEncoder[Y, X]":
        r"""Return the inverse encoder (i.e. decoder).

        Example:
            >>> enc = ~self
            >>> enc.encode(y) == self.decode(y)
        """
        return InverseEncoder(self)

    def __matmul__[T](self, other: Encoder[T, X], /) -> "ChainedEncoder[T, Y]":
        r"""Chain the encoders (pure function composition).

        Example:
            >>> enc = self @ other
            >>> enc.encode(0) == self.encode(other.encode(0))
        """
        return ChainedEncoder(self, other)

    def __rmatmul__[Z](self, other: Encoder[Y, Z], /) -> "ChainedEncoder[X, Z]":
        r"""Chain the encoders (pure function composition).

        See `__matmul__` for more details.
        """
        return ChainedEncoder(other, self)

    def __rshift__[Z](self, other: Encoder[Y, Z], /) -> "PipedEncoder[X, Z]":
        r"""Pipe the encoders (encoder composition).

        Note that the order is reversed compared to the `@`-operator.

        Example:
            >>> enc1, enc2, x = ...
            >>> enc = enc1 >> enc2
            >>> assert (y := enc(x)) == enc2(enc1(x))
            >>> assert enc.encode(x) == enc2.encode(enc1.encode(x))
            >>> assert enc.decode(y) == enc1.decode(enc2.decode(y))

        Note:
            - `>>` is associative: `(A >> B) >> C = A >> (B >> C)`

              .. math::
                 ((A >> B) >> C)(x) = C((A >> B)(x)) = C(B(A(x)))  \\
                 (A >> (B >> C))(x) = (B >> C)(A(x)) = C(B(A(x)))

            .. details:: inverse law: `~(A >> B) == ~B >> ~A`

                      ~(A >> B)(x)
                          = (A >> B).decode(x)
                          = B.decode(A.decode(x))
                          = ~B(~A(x))
                          = (~B >> ~A)(x)
        """
        return PipedEncoder(self, other)

    def __rrshift__[T](self, other: Encoder[T, X], /) -> "PipedEncoder[T, Y]":
        r"""Pipe the encoders (encoder composition).

        See `__rshift__` for more details.
        """
        return PipedEncoder(other, self)

    def __or__[X2, Y2](self, other: Encoder[X2, Y2], /) -> "ParallelEncoder[tuple[X, X2], tuple[Y, Y2]]":  # fmt: skip
        r"""Return product encoders.

        Example:
            >>> enc = self | other
            >>> enc((x, y)) == (self(x), other(y))
        """
        return ParallelEncoder(self, other)

    def __ror__[X2, Y2](self, other: Encoder[X2, Y2], /) -> "ParallelEncoder[tuple[X2, X], tuple[Y2, Y]]":  # fmt: skip
        r"""Return product encoders.

        See `__or__` for more details.
        """
        return ParallelEncoder(other, self)

    def __and__[Y2](self, other: Encoder[X, Y2], /) -> "JointEncoder[X, tuple[Y, Y2]]":
        r"""Return joint encoders.

        Example:
            >>> enc = self & other
            >>> enc(x) == (self(x), other(x))
        """
        # FIXME: mypy does not predict correct return type...
        return JointEncoder(self, other)

    def __rand__[Y2](self, other: Encoder[X, Y2], /) -> "JointEncoder[X, tuple[Y2, Y]]":
        r"""Return joint encoders.

        See `__and__` for more details.
        """
        # FIXME: mypy does not predict correct return type...
        return JointEncoder(other, self)

    # endregion magic methods ----------------------------------------------------------

    # region chaining methods ----------------------------------------------------------
    def standardize(self) -> "BaseEncoder[X, Y]":
        r"""Chain a standardizer."""
        return self >> E.StandardScaler()

    def minmax_scale(self) -> "BaseEncoder[X, Y]":
        r"""Chain a minmax scaling."""
        return self >> E.MinMaxScaler()

    # endregion chaining methods -------------------------------------------------------


class UniversalEncoder(BaseEncoder[Any, Any], ABC):
    r"""Encoder class which maps data to the same type, regardless of the input."""

    @abstractmethod
    def encode[T](self, x: T, /) -> T: ...
    @abstractmethod
    def decode[T](self, y: T, /) -> T: ...
    def fit(self, data: Any, /) -> None: ...


@pprint_sequence(recursive=2)
class EncoderList[X, Y](BaseEncoder[X, Y], Sequence[Encoder]):
    r"""Wraps List of encoders."""

    encoders: list[Encoder]
    r"""List of encoders."""

    def __init__(self, *encoders: Encoder) -> None:
        r"""Initialize the encoder list."""
        self.encoders = list(encoders)

    @property
    def params(self) -> dict[str, Any]:
        return {"encoders": self.encoders}

    @property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self.encoders)

    @property
    def is_fitted(self) -> bool:
        return all(e.is_fitted for e in self.encoders)

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        for encoder in self.encoders:
            encoder.is_fitted = value

    def __len__(self) -> int:
        r"""Return number of chained encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> Encoder: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    def __getitem__(self, index):
        r"""Get the encoder at the given index."""
        match index:
            case int(idx):
                return self.encoders[idx]
            case slice() as slc:
                cls = type(self)
                return cls(*self.encoders[slc])
            case _:
                raise TypeError(f"Type {type(index)} not supported.")

    def simplify(self) -> "Encoder[X, Y]":
        r"""Simplify the encoder."""
        cls = type(self)
        return cls(*(e.simplify() for e in self))


@pprint_mapping(recursive=2)
@dataclass(init=False)
class EncoderDict[X, Y, K](BaseEncoder[X, Y], Mapping[K, Encoder]):
    r"""Wraps dictionary of encoders."""

    encoders: dict[K, Encoder]
    r"""Mapping of keys to encoders."""

    @overload
    def __init__(
        self,
        enc_map: SupportsKeysAndGetItem[K, Encoder],
        /,
    ) -> None: ...
    @overload
    def __init__[U, V](
        self: "EncoderDict[U, V, str]",
        enc_map: SupportsKeysAndGetItem[str, Encoder] = ...,
        /,
        **encoders: Encoder,
    ) -> None: ...
    def __init__(
        self,
        enc_map: SupportsKeysAndGetItem[Any, Encoder] = EMPTY_MAP,
        /,
        **encoders: Encoder,
    ) -> None:
        self.encoders = dict(enc_map, **encoders)

    @property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self.encoders.values())

    @property
    def is_fitted(self) -> bool:
        return all(e.is_fitted for e in self.encoders.values())

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        for encoder in self.encoders.values():
            encoder.is_fitted = value

    def __len__(self) -> int:
        return len(self.encoders)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.encoders)

    def __getitem__(self, key: Any, /) -> Encoder:
        r"""Get the encoder for the given key."""
        return self.encoders[key]

    def simplify(self) -> "IdentityEncoder | Self":
        r"""Simplify the mapping encoder."""
        # FIXME: https://github.com/python/mypy/issues/17134
        # match self:
        #     case []:
        #         return IdentityEncoder()
        #     case [encoder]:
        #         return encoder.simplify()
        #     case _:
        #         cls = type(self)
        #         return cls(*(e.simplify() for e in self))

        if len(self.encoders) == 0:
            return IdentityEncoder()
        cls = type(self)
        return cls({k: e.simplify() for k, e in self.encoders.items()})


class BackendMixin:
    r"""Encoder equipped with a backend."""

    def __init_subclass__(cls: type[Encoder]) -> None:
        super().__init_subclass__()

        original_fit = cls.fit

        @wraps(original_fit)
        def wrapped_fit(self, x, /):
            if self.backend is NotImplemented:
                self.backend = get_backend(x)
            original_fit(self, x)

        cls.fit = wrapped_fit

    backend: Backend = NotImplemented


# endregion base classes ---------------------------------------------------------------


# region nullary encoders --------------------------------------------------------------
class IdentityEncoder(BaseEncoder[Any, Any]):
    r"""Dummy class that performs identity function."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def encode[T](self, x: T, /) -> T:
        return x

    def decode[T](self, y: T, /) -> T:
        return y


class DeepcopyEncoder(BaseEncoder[Any, Any]):
    r"""Encoder that deepcopies the input."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def encode[T](self, x: T, /) -> T:
        return deepcopy(x)

    def decode[T](self, y: T, /) -> T:
        return deepcopy(y)


class TupleEncoder(BaseEncoder[Any, Any]):
    r"""Wraps input into a tuple."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def __invert__(self) -> "TupleDecoder":
        return TupleDecoder()

    def encode[T](self, x: T, /) -> tuple[T]:
        return (x,)

    def decode[T](self, y: tuple[T], /) -> T:
        return y[0]


class TupleDecoder(BaseEncoder[Any, Any]):
    r"""Unwraps input from a tuple."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def __invert__(self) -> "TupleEncoder":
        return TupleEncoder()

    def encode[T](self, y: tuple[T], /) -> T:
        return y[0]

    def decode[T](self, x: T, /) -> tuple[T]:
        return (x,)


@dataclass
class DiagonalEncoder[T](BaseEncoder[T, tuple[T, ...]]):
    r"""Encodes the input into a tuple of itself.

    .. math:: f(x) = (x, x, …, x)

    Note:
        In practice, when working with float arrays, we need to be careful how to select
        the inverse. Due to rounding errors, the values in the tuple elements might be
        slightly different. In this case, an aggregation function needs to be supplied.
    """

    num: int

    _: KW_ONLY

    aggregate_fn: Callable[[tuple[T, ...]], T] = random.choice

    def __invert__(self) -> "DiagonalDecoder":
        return DiagonalDecoder(num=self.num, aggregate_fn=self.aggregate_fn)

    def encode(self, x: T, /) -> tuple[T, ...]:
        return (x,) * self.num

    def decode(self, y: tuple[T, ...], /) -> T:
        return self.aggregate_fn(y)


@dataclass
class DiagonalDecoder[T](BaseEncoder[tuple[T, ...], T]):
    r"""Encodes the input into a tuple of itself.

    .. math:: f(x) = (x, x, …, x)

    Note:
        In practice, when working with float arrays, we need to be careful how to select
        the inverse. Due to rounding errors, the values in the tuple elements might be
        slightly different. In this case, an aggregation function needs to be supplied.
    """

    num: int

    _: KW_ONLY

    aggregate_fn: Callable[[tuple[T, ...]], T] = random.choice

    def __invert__(self) -> "DiagonalEncoder":
        return DiagonalEncoder(num=self.num, aggregate_fn=self.aggregate_fn)

    def encode(self, y: tuple[T, ...], /) -> T:
        return self.aggregate_fn(y)

    def decode(self, x: T, /) -> tuple[T, ...]:
        return (x,) * self.num


# endregion nullary encoders -----------------------------------------------------------


# region unary encoders ----------------------------------------------------------------
@dataclass(frozen=True, slots=True, repr=False)
class InverseEncoder[X, Y](BaseEncoder[Y, X]):
    r"""Applies an encoder in reverse."""

    encoder: Encoder[X, Y]
    r"""The encoder to invert."""

    def fit(self, y: Y, /) -> None:
        raise NotImplementedError("Inverse encoders cannot be fitted.")

    def encode(self, y: Y, /) -> X:
        return self.encoder.decode(y)

    def decode(self, x: X, /) -> Y:
        return self.encoder.encode(x)

    def simplify(self) -> Self:
        cls = type(self)
        return cls(self.encoder.simplify())

    def __repr__(self) -> str:
        return f"~{self.encoder}"


def invert_encoder[X, Y](encoder: Encoder[X, Y], /, *, simplify: bool = True) -> Encoder[Y, X]:  # fmt: skip
    r"""Return the inverse encoder (i.e. decoder)."""
    decoder = InverseEncoder(encoder)
    return decoder.simplify() if simplify else decoder


@pprint_repr
@dataclass
class WrappedEncoder[X, Y](BaseEncoder[X, Y]):
    r"""Wraps an encoder."""

    encoder: Encoder[X, Y]
    r"""The encoder to wrap."""

    def __invert__(self) -> "WrappedEncoder[Y, X]":
        # FIXME: https://github.com/microsoft/pyright/issues/8165
        decoder = invert_encoder(self.encoder)
        return WrappedEncoder(decoder)

    @property
    def params(self) -> dict[str, Any]:
        return self.encoder.params

    def encode(self, x: X, /) -> Y:
        return self.encoder.encode(x)

    def decode(self, y: Y, /) -> X:
        return self.encoder.decode(y)


@pprint_repr
@dataclass
class NestedEncoder[X, Y](BaseEncoder[NestedBuiltin[X], NestedBuiltin[Y]]):
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

    leaf_type: type[X] = object  # type: ignore[assignment]
    r"""The type of the leaf elements."""
    output_leaf_type: type[Y] = object  # type: ignore[assignment]
    r"""The type of the output elements."""

    def __invert__(self) -> "NestedEncoder[Y, X]":
        # FIXME: https://github.com/microsoft/pyright/issues/8165
        decoder = invert_encoder(self.encoder)
        cls: type[NestedEncoder] = type(self)
        return cls(
            decoder,
            leaf_type=self.output_leaf_type,
            output_leaf_type=self.leaf_type,
        )

    def encode(self, x: NestedBuiltin[X], /) -> NestedBuiltin[Y]:
        match x:
            case list(seq):
                return [self.encode(val) for val in seq]
            case tuple(tup):
                return tuple(self.encode(val) for val in tup)
            case dict(mapping):
                return {key: self.encode(val) for key, val in mapping.items()}
            case set(items):
                return {self.encode(val) for val in items}  # pyright: ignore[reportUnhashable]
            case frozenset(items):
                return frozenset(self.encode(val) for val in items)
            case self.leaf_type() as leaf:  # pyright: ignore[reportGeneralTypeIssues]
                return self.encoder.encode(leaf)
            case _:
                raise TypeError(f"Type {type(x)} not supported.")

    def decode(self, y: NestedBuiltin[Y], /) -> NestedBuiltin[X]:
        match y:
            case list(seq):
                return [self.decode(val) for val in seq]
            case tuple(tup):
                return tuple(self.decode(val) for val in tup)
            case dict(mapping):
                return {key: self.decode(val) for key, val in mapping.items()}
            case set(items):
                return {self.decode(val) for val in items}  # pyright: ignore[reportUnhashable]
            case frozenset(items):
                return frozenset(self.decode(val) for val in items)
            case self.output_leaf_type() as leaf:  # pyright: ignore[reportGeneralTypeIssues]
                return self.encoder.decode(leaf)
            case _:
                raise TypeError(f"Type {type(y)} not supported.")


# endregion unary encoders -------------------------------------------------------------


# region variadic encoders -------------------------------------------------------------
@pprint_repr(recursive=2)
class ChainedEncoder[X, Y](EncoderList[X, Y]):
    r"""Represents function composition of encoders."""

    # fmt: off
    @overload  # n=0
    def __new__(cls, *encoders: *tuple[()]) -> Self: ...
    @overload  # n=1
    def __new__(cls, *encoders: *tuple[Encoder[X, Y]]) -> Self: ...
    @overload  # n=2
    def __new__[Z](cls, *encoders: *tuple[Encoder[Z, Y], Encoder[X, Z]]) -> Self: ...
    @overload  # n>2
    def __new__(cls, *encoders: *tuple[Encoder[Any, Y], *tuple[Encoder, ...], Encoder[X, Any]]) -> Self: ...
    # fmt: on
    def __new__(cls, *encoders):
        r"""Create a new chained encoder."""
        return super().__new__(cls)

    def __invert__(self) -> "ChainedEncoder[Y, X]":
        cls: type[ChainedEncoder] = type(self)
        return cls(*(InverseEncoder(e) for e in reversed(self.encoders)))

    def fit(self, x: X, /) -> None:
        for encoder in reversed(self.encoders):
            try:
                encoder.fit(x)
            except Exception as exc:
                index = self.encoders.index(encoder)
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{index}]: Failed to fit {enc!r}.")
                raise
            else:
                x = encoder.encode(x)

    def encode(self, x: X, /) -> Y:
        for encoder in reversed(self.encoders):
            x = encoder.encode(x)
        return cast(Y, x)

    def decode(self, y: Y, /) -> X:
        for encoder in self.encoders:
            y = encoder.decode(y)
        return cast(X, y)

    def simplify(self) -> IdentityEncoder | Encoder[X, Y] | Self:
        r"""Simplify the chained encoder."""
        # simplify the nested encoders
        encoders: list[Encoder] = []
        for encoder in (e.simplify() for e in self):
            match encoder:
                case PipedEncoder(encoders=nested):
                    encoders.extend(reversed(nested))
                case ChainedEncoder(encoders=nested):
                    encoders.extend(nested)
                case _:
                    encoders.append(encoder)

        # simplify self
        match encoders:
            case []:
                return IdentityEncoder()
            case [encoder]:
                return encoder
            case _:
                cls = type(self)
                return cls(*encoders)


# fmt: off
# # simplify: Literal[False]
# @overload  # n=0, simplify=False
# def chain_encoders(*, simplify: Literal[False]) -> ChainedEncoder[Any, Any]: ...
# @overload  # n=1, simplify=False
# def chain_encoders(e: Encoder[X, Y], /, *, simplify: Literal[False]) -> ChainedEncoder[X, Y]: ...
# @overload  # n=2, simplify=False
# def chain_encoders(e1: Encoder[Y, Z], e2: Encoder[X, Y], /, *, simplify: Literal[False]) -> ChainedEncoder[X, Z]: ...
# @overload  # n>2, simplify=False
# def chain_encoders(
#   *es: *tuple[Encoder[Any, Y], *tuple[Encoder, ...], Encoder[X, Any]], simplify: Literal[False]
# ) -> ChainedEncoder[X, Y]: ...
# # simplify: Literal[True]
# @overload  # n=0, simplify=False
# def chain_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
# @overload  # n=1, simplify=False
# def chain_encoders(e: Encoder[X, Y], /, *, simplify: Literal[True]) -> Encoder[X, Y]: ...
# @overload  # n=2, simplify=False
# def chain_encoders(e1: Encoder[Y, Z], e2: Encoder[X, Y], /, *, simplify: Literal[True]) -> Encoder[X, Z]: ...
# @overload  # n>2, simplify=False
# def chain_encoders(
#   *es: *tuple[Encoder[Any, Y], *tuple[Encoder, ...], Encoder[X, Any]], simplify: Literal[True]
# ) -> Encoder[X, Y]: ...
# simplify: bool = ...
@overload  # n=0, simplify: bool
def chain_encoders(*, simplify: bool = ...) -> Encoder[Any, Any]: ...
@overload  # n=1, simplify: bool
def chain_encoders[X, Y](e: Encoder[X, Y], /, *, simplify: bool = ...) -> Encoder[X, Y]: ...
@overload  # n=2, simplify: bool
def chain_encoders[X, Y, Z](e1: Encoder[Y, Z], e2: Encoder[X, Y], /, *, simplify: bool = ...) -> Encoder[X, Z]: ...
@overload  # n>2, simplify: bool
def chain_encoders[X, Y](*es: *tuple[Encoder[Any, Y], *tuple[Encoder, ...], Encoder[X, Any]], simplify: bool = ...) -> Encoder[X, Y]: ...
# fmt: on
def chain_encoders(*encoders: Encoder, simplify: bool = True) -> Encoder:  # type: ignore[misc]
    r"""Chain encoders."""
    encoder = ChainedEncoder(*encoders)
    return encoder.simplify() if simplify else encoder


@pprint_repr(recursive=2)
class PipedEncoder[X, Y](EncoderList[X, Y]):
    r"""Represents function composition of encoders."""

    # fmt: off
    @overload  # n=0
    def __new__(cls, *encoders: *tuple[()]) -> Self: ...
    @overload  # n=1
    def __new__(cls, *encoders: *tuple[Encoder[X, Y]]) -> Self: ...
    @overload  # n=2
    def __new__[Z](cls, *encoders: *tuple[Encoder[X, Z], Encoder[Z, Y]]) -> Self: ...
    @overload  # n>2
    def __new__(cls, *encoders: *tuple[Encoder[X, Any], *tuple[Encoder, ...], Encoder[Any, Y]]) -> Self: ...
    # fmt: on
    def __new__(cls, *encoders):
        r"""Create a new chained encoder."""
        return super().__new__(cls)

    def __invert__(self) -> "PipedEncoder[Y, X]":
        cls: type[PipedEncoder] = type(self)
        return cls(*(InverseEncoder(e) for e in reversed(self.encoders)))

    def fit(self, x: X, /) -> None:
        for encoder in self.encoders:
            try:
                encoder.fit(x)
            except Exception as exc:
                index = self.encoders.index(encoder)
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{index}]: Failed to fit {enc!r}.")
                raise
            else:
                x = encoder.encode(x)

    def encode(self, x: X, /) -> Y:
        for encoder in self.encoders:
            x = encoder.encode(x)
        return cast(Y, x)

    def decode(self, y: Y, /) -> X:
        for encoder in reversed(self.encoders):
            y = encoder.decode(y)
        return cast(X, y)

    def simplify(self) -> IdentityEncoder | Encoder[X, Y] | Self:
        r"""Simplify the chained encoder."""
        # recursively simplify the nested encoders
        encoders = [e.simplify() for e in self]

        # reduction 1: combine nested pipes/chains `e >> (f >> g) = (e >> f) >> g`.
        es = []
        for encoder in encoders:
            match encoder:
                case PipedEncoder(encoders=nested):
                    es.extend(nested)
                case ChainedEncoder(encoders=nested):
                    es.extend(reversed(nested))
                case _:
                    es.append(encoder)

        # reduction 2: remove identity encoders `e >> id = e`.
        es = [e for e in es if not isinstance(e, IdentityEncoder)]

        # reduction 3: remove inverse pairs `(e >> ~e) = id`.

        # reduction 4: remove idempotent encoders `e >> e = e`.

        # simplify self
        match es:
            case []:
                return IdentityEncoder()
            case [encoder]:
                return encoder
            case _:
                cls = type(self)
                return cls(*es)


# fmt: off
# # simplify: Literal[False]
# @overload  # n=0, simplify=False
# def pipe_encoders(*, simplify: Literal[False]) -> PipedEncoder[Any, Any]: ...
# @overload  # n=1, simplify=False
# def pipe_encoders(e: Encoder[X, Y], /, *, simplify: Literal[False]) -> PipedEncoder[X, Y]: ...
# @overload  # n=2, simplify=False
# def pipe_encoders(e1: Encoder[X, Y], e2: Encoder[Y, Z], /, *, simplify: Literal[False]) -> PipedEncoder[X, Z]: ...
# @overload  # n>2, simplify=False
# def pipe_encoders(
#   *es: *tuple[Encoder[X, Any], *tuple[Encoder, ...], Encoder[Any, Y]], simplify: Literal[False]
# ) -> PipedEncoder[X, Y]: ...
# # simplify: Literal[True]
# @overload  # n=0, simplify=True
# def pipe_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
# @overload  # n=1, simplify=True
# def pipe_encoders(e: Encoder[X, Y], /, *, simplify: Literal[True]) -> Encoder[X, Y]: ...
# @overload  # n=2, simplify=True
# def pipe_encoders(e1: Encoder[X, Y], e2: Encoder[Y, Z], /, *, simplify: Literal[True]) -> Encoder[X, Z]: ...
# @overload  # n>2, simplify=True
# def pipe_encoders(
#   *es: *tuple[Encoder[X, Any], *tuple[Encoder, ...], Encoder[Any, Y]], simplify: Literal[True]
# ) -> Encoder[X, Y]: ...
# simplify: bool = ...
@overload  # n=0
def pipe_encoders(*, simplify: bool = ...) -> Encoder[Any, Any]: ...
@overload  # n=1
def pipe_encoders[X, Y](e: Encoder[X, Y], /, *, simplify: bool = ...) -> Encoder[X, Y]: ...
@overload  # n=2
def pipe_encoders[X, Y, Z](e1: Encoder[X, Y], e2: Encoder[Y, Z], /, *, simplify: bool = ...) -> Encoder[X, Z]: ...
@overload  # n>2
def pipe_encoders[X, Y](*es: *tuple[Encoder[X, Any], *tuple[Encoder, ...], Encoder[Any, Y]], simplify: bool = ...) -> Encoder[X, Y]: ...
# fmt: on
def pipe_encoders(*encoders: Encoder, simplify: bool = True) -> Encoder:  # type: ignore[misc]
    r"""Pipe encoders."""
    encoder = PipedEncoder(*encoders)
    return encoder.simplify() if simplify else encoder


# fmt: off
@overload  # n=-1
def pow_encoder[X, Y](e: Encoder[X, Y], n: Literal[-1], /, *, simplify: Literal[True] = ..., copy: bool = ...) -> Encoder[Y, X]: ...
@overload  # n=0
def pow_encoder(e: Encoder, n: Literal[0], /, *, simplify: Literal[True] = ..., copy: bool = ...) -> IdentityEncoder: ...
@overload  # n=1
def pow_encoder[X, Y](e: Encoder[X, Y], n: Literal[1], /, *, simplify: Literal[True] = ..., copy: bool = ...) -> Encoder[X, Y]: ...
@overload  # n>1
def pow_encoder[T](e: Encoder[T, T], n: int, /, *, simplify: bool = ..., copy: bool = ...) -> PipedEncoder[T, T]: ...
# fmt: on
def pow_encoder(encoder, n, /, *, simplify=True, copy=True):
    r"""Apply encoder n times."""
    encoder = encoder.simplify() if simplify else encoder
    encoders = [(deepcopy(encoder) if copy else encoder) for _ in range(n)]

    if n == -1 and simplify:
        return ~encoders[0]
    if n == 0 and simplify:
        return IdentityEncoder()
    if n == 1 and simplify:
        return encoders[0]
    if n < 0:
        return pipe_encoders(*[~e for e in reversed(encoders)])
    return PipedEncoder(*encoders)


# FIXME: We could have better type hints with HKTs
@pprint_repr(recursive=2)
class ParallelEncoder[TupleIn: tuple, TupleOut: tuple](EncoderList[TupleIn, TupleOut]):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """

    # fmt: off
    @overload  # n=0
    def __new__(cls, *encoders: *tuple[()]) -> "ParallelEncoder[Any, Any]": ...
    @overload  # n=1
    def __new__[X, Y](cls, *encoders: *tuple[Encoder[X, Y]]) -> "ParallelEncoder[tuple[X], tuple[Y]]": ...
    @overload  # n=2
    def __new__[X1, Y1, X2, Y2](cls, *encoders: *tuple[Encoder[X1, Y1], Encoder[X2, Y2]]) -> "ParallelEncoder[tuple[X1, X2], tuple[Y1, Y2]]": ...
    @overload  # n>2
    def __new__[X, Y](cls, *encoders: Encoder[X, Y]) -> "ParallelEncoder[tuple[X, ...], tuple[Y, ...]]": ...
    # fmt: on
    # NOTE: We need to ignore inconsistent overload errors because `Self` cannot be generic.
    def __new__(cls, *encoders):  # pyright: ignore[reportInconsistentOverload]
        r"""Create a new chained encoder."""
        return super().__new__(cls)

    def __invert__(self) -> "ParallelEncoder[TupleOut, TupleIn]":
        cls: type[ParallelEncoder] = type(self)
        return cls(*(InverseEncoder(e) for e in self.encoders))  # type: ignore[return-value]

    def fit(self, xs: TupleIn, /) -> None:
        for encoder, x in zip(self.encoders, xs, strict=True):
            encoder.fit(x)

    def encode(self, xs: TupleIn, /) -> TupleOut:
        return tuple(  # type: ignore[return-value]
            encoder.encode(x) for encoder, x in zip(self.encoders, xs, strict=True)
        )

    def decode(self, ys: TupleOut, /) -> TupleIn:
        return tuple(  # type: ignore[return-value]
            encoder.decode(x) for encoder, x in zip(self.encoders, ys, strict=True)
        )

    def simplify(self) -> IdentityEncoder | Encoder | Self:
        r"""Simplify the product encoder."""
        # FIXME: https://github.com/python/mypy/issues/17134
        # match self:
        #     case []:
        #         return IdentityEncoder()
        #     case [encoder]:
        #         return encoder.simplify()
        #     case _:
        #         cls = type(self)
        #         return cls(*(e.simplify() for e in self))

        if len(self.encoders) == 0:
            return IdentityEncoder()
        if len(self.encoders) == 1:
            encoder = self.encoders[0].simplify()
            return TupleDecoder() >> encoder >> TupleEncoder()
        cls = type(self)
        return cls(*(e.simplify() for e in self.encoders))


# fmt: off
@overload  # n=0
def parallelize_encoders(*, simplify: bool = ...) -> Encoder[Any, Any]: ...
@overload  # n=1
def parallelize_encoders[X, Y](e: Encoder[X, Y], /, *, simplify: bool = ...) -> Encoder[tuple[X], tuple[Y]]: ...
@overload  # n=2
def parallelize_encoders[X1, Y1, X2, Y2](e1: Encoder[X1, Y1], e2: Encoder[X2, Y2], /, *, simplify: bool = ...) -> Encoder[tuple[X1, X2], tuple[Y1, Y2]]: ...
@overload  # n>2
def parallelize_encoders[X, Y](*encoders: Encoder[X, Y], simplify: bool = ...) -> Encoder[tuple[X, ...], tuple[Y, ...]]: ...
@overload  # fallback
def parallelize_encoders(*encoders: Encoder, simplify: bool = ...) -> Encoder: ...
# fmt: on
def parallelize_encoders(*encoders, simplify=True):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """
    encoder = ParallelEncoder(*encoders)
    return encoder.simplify() if simplify else encoder


# fmt: off
@overload  # n=0
def duplicate_encoder(e: Encoder, n: Literal[0], /, *, simplify: Literal[True], copy: bool = ...) -> IdentityEncoder: ...
@overload  # n=1
def duplicate_encoder(e: Encoder, n: Literal[1], /, *, simplify: Literal[True], copy: bool = ...) -> Encoder: ...
@overload  # n>1
def duplicate_encoder(e: Encoder, n: int, /, *, simplify: bool = ..., copy: bool = ...) -> ParallelEncoder: ...
# fmt: on
def duplicate_encoder(encoder, n, /, *, simplify=True, copy=True):
    r"""Duplicate an encoder."""
    encoder = encoder.simplify() if simplify else encoder
    encoders = [deepcopy(encoder) if copy else encoder for _ in range(n)]

    if n == -1 and simplify:
        return ~encoders[0]
    if n == 0 and simplify:
        return IdentityEncoder()
    if n == 1 and simplify:
        return encoders[0]
    if n < 0:
        return parallelize_encoders(*(~e for e in reversed(encoders)))
    return parallelize_encoders(*encoders)


@pprint_repr(recursive=2)
class JointEncoder[X, TupleOut: tuple](EncoderList[X, TupleOut]):
    r"""Factorized Encoder.

    Example:
        >>> enc = FactorizedEncoder(e1, e2, e3)
        >>> enc(x) == (e1(x), e2(x), e3(x))

    Note:
        This is essentially equivalent to chaining `DiagonalEncoder >> ParallelEncoder`.
    """

    encoders: list[Encoder[X, Any]]
    aggregate_fn: Agg[X] = random.choice

    @property
    def params(self):
        return {"encoders": self.encoders, "aggregate_fn": self.aggregate_fn}

    # fmt: off
    @overload  # n=0
    def __new__(cls, *, aggregate_fn:  Agg[X] = ...) -> "JointEncoder[X, tuple[()]]": ...
    @overload  # n=1
    def __new__[Y](cls, e: Encoder[X, Y], /, *, aggregate_fn:  Agg[X] = ...) -> "JointEncoder[X, tuple[Y]]": ...
    @overload  # n=2
    def __new__[Y1, Y2](cls, e1: Encoder[X, Y1], e2: Encoder[X, Y2], /, *, aggregate_fn:  Agg[X] = ...) -> "JointEncoder[X, tuple[Y1, Y2]]": ...
    @overload  # n>2
    def __new__[Y](cls, *es: Encoder[X, Y], aggregate_fn: Agg[X] = ...) -> "JointEncoder[X, tuple[Y, ...]]": ...
    # fmt: on
    # NOTE: We need to ignore inconsistent overload errors because `Self` cannot be generic.
    def __new__(  # type: ignore[misc]
        cls, *encoders: Encoder[X, Any], aggregate_fn: Agg[X] = random.choice
    ) -> Self:
        r"""Create a new chained encoder."""
        return super().__new__(cls)

    # NOTE: Need to use different variable names than the class-scoped parameters!
    # fmt: off
    @overload  # n=0
    def __init__[T](self: "JointEncoder[T, tuple[()]]", *, aggregate_fn: Agg[T] = ...) -> None: ...
    @overload  # n=1
    def __init__[T, Y](self: "JointEncoder[T, tuple[Y]]", e: Encoder[T, Y], /, *, aggregate_fn: Agg[T] = ...) -> None: ...
    @overload  # n=2
    def __init__[T, Y1, Y2](self: "JointEncoder[T, tuple[Y1, Y2]]", e1: Encoder[T, Y1], e2: Encoder[T, Y2], /, *, aggregate_fn: Agg[T] = ...) -> None: ...
    @overload  # n>2
    def __init__[T, Y](self: "JointEncoder[T, tuple[Y, ...]]", *es: Encoder[T, Y], aggregate_fn: Agg[T] = ...) -> None: ...
    # fmt: on
    def __init__(self, *encoders: Encoder, aggregate_fn: Agg = random.choice) -> None:
        super().__init__(*encoders)
        self.aggregate_fn = aggregate_fn

    def __invert__(self) -> "JointDecoder[TupleOut, X]":
        # Q: Why does pyright error here?
        decoders = (InverseEncoder(e) for e in self.encoders)
        return JointDecoder(*decoders, aggregate_fn=self.aggregate_fn)  # type: ignore[return-value]

    def encode(self, x: X, /) -> TupleOut:
        return tuple(e.encode(x) for e in self.encoders)  # type: ignore[return-value]

    def decode(self, ys: TupleOut, /) -> X:
        decoded_vals = [e.decode(y) for e, y in zip(self.encoders, ys, strict=True)]
        return self.aggregate_fn(decoded_vals)

    def simplify(self) -> IdentityEncoder | Encoder[X, TupleOut] | Self:
        r"""Simplify the joint encoder."""
        # FIXME: https://github.com/python/mypy/issues/17134
        if len(self) == 0:
            return IdentityEncoder()
        if len(self) == 1:
            return (self[0] >> TupleEncoder()).simplify()
        cls = type(self)
        return cls(*(e.simplify() for e in self))


# fmt: off
@overload  # n=0
def join_encoders(*, aggregate_fn: Agg = ..., simplify: bool = ...) -> Encoder[Any, Any]: ...
@overload  # n=1
def join_encoders[X, Y](e: Encoder[X, Y], /, *, aggregate_fn: Agg[X] = ..., simplify: bool = ...) -> Encoder[X, tuple[Y]]: ...
@overload  # n=2
def join_encoders[X, Y1, Y2](e1: Encoder[X, Y1], e2: Encoder[X, Y2], /, *, aggregate_fn: Agg[X] = ..., simplify: bool = ...) -> Encoder[X, tuple[Y1, Y2]]: ...
@overload  # n>2
def join_encoders[X, Y](*es: Encoder[X, Y], aggregate_fn: Agg[X] = ..., simplify: bool = ...) -> Encoder[X, tuple[Y, ...]]: ...
# fmt: on
def join_encoders[X, Y](  # type: ignore[misc]
    *encoders: Encoder[X, Y],
    aggregate_fn: Agg[X] = random.choice,
    simplify: bool = True,
) -> Encoder[X, tuple[Y, ...]]:
    r"""Join encoders."""
    encoder = JointEncoder(*encoders, aggregate_fn=aggregate_fn)
    return encoder.simplify() if simplify else encoder


@pprint_repr(recursive=2)
class JointDecoder[TupleIn: tuple, Y](EncoderList[TupleIn, Y]):
    r"""Factorized Encoder.

    Example:
        >>> enc = FactorizedEncoder(e1, e2, e3)
        >>> enc(x) == (e1(x), e2(x), e3(x))

    Note:
        This is essentially equivalent to chaining `DiagonalEncoder >> ParallelEncoder`.
    """

    encoders: list[Encoder[Any, Y]]
    aggregate_fn: Agg[Y] = random.choice

    @property
    def params(self):
        return {"encoders": self.encoders, "aggregate_fn": self.aggregate_fn}

    # fmt: off
    @overload  # n=0
    def __new__(cls, *, aggregate_fn: Agg[Y] = ...) -> "JointDecoder[tuple[()], Y]": ...
    @overload  # n=1
    def __new__[X](cls, e: Encoder[X, Y], /, *, aggregate_fn: Agg[Y] = ...) -> "JointDecoder[tuple[X], Y]": ...
    @overload  # n=2
    def __new__[X1, X2](cls, e1: Encoder[X1, Y], e2: Encoder[X2, Y], /, *, aggregate_fn: Agg[Y] = ...) -> "JointDecoder[tuple[X1, X2], Y]": ...
    @overload  # n>2
    def __new__[X](cls, *es: Encoder[X, Y], aggregate_fn: Agg[Y] = ...) -> "JointDecoder[tuple[X, ...], Y]": ...
    # fmt: on
    # NOTE: We need to ignore inconsistent overload errors because `Self` cannot be generic.
    def __new__(  # type: ignore[misc]
        cls,
        *encoders: Encoder[Any, Y],
        aggregate_fn: Agg[Y] = random.choice,
    ) -> Self:
        r"""Create a new chained encoder."""
        return super().__new__(cls)

    # NOTE: Need to use different variable names than the class-scoped parameters!
    # fmt: off
    @overload  # n=0
    def __init__[Z](self: "JointDecoder[tuple[()], Z]", *, aggregate_fn: Agg[Z] = ...) -> None: ...
    @overload  # n=1
    def __init__[X, Z](self: "JointDecoder[tuple[X], Z]", e: Encoder[X, Z], /, *, aggregate_fn: Agg[Z] = ...) -> None: ...
    @overload  # n=2
    def __init__[X1, X2, Z](self: "JointDecoder[tuple[X1, X2], Z]", e1: Encoder[X1, Z], e2: Encoder[X2, Z], /, *, aggregate_fn: Agg[Z] = ...) -> None: ...
    @overload  # n>2
    def __init__[X, Z](self: "JointDecoder[tuple[X, ...], Z]", *es: Encoder[X, Z], aggregate_fn: Agg[Z] = ...) -> None: ...
    # fmt: on
    def __init__(self, *encoders: Encoder, aggregate_fn: Agg = random.choice) -> None:
        super().__init__(*encoders)
        self.aggregate_fn = aggregate_fn

    def __invert__(self) -> "JointEncoder[Y, TupleIn]":
        decoders = (InverseEncoder(e) for e in self.encoders)
        return JointEncoder(*decoders, aggregate_fn=self.aggregate_fn)  # type: ignore[return-value]

    def encode(self, xs: TupleIn, /) -> Y:
        encoded_vals = [e.encode(x) for e, x in zip(self.encoders, xs, strict=True)]
        return self.aggregate_fn(encoded_vals)

    def decode(self, y: Y, /) -> TupleIn:
        return tuple(e.decode(y) for e in self.encoders)  # type: ignore[return-value]

    def simplify(self) -> IdentityEncoder | Encoder[TupleIn, Y] | Self:
        r"""Simplify the joint encoder."""
        # FIXME: https://github.com/python/mypy/issues/17134
        if len(self) == 0:
            return IdentityEncoder()
        if len(self) == 1:
            return (self[0] >> TupleEncoder()).simplify()
        cls = type(self)
        return cls(*(e.simplify() for e in self))


class MappedEncoder[
    MappingIn: Mapping,
    MappingOut: Mapping,
    K,
](EncoderDict[MappingIn, MappingOut, K]):
    r"""Maps encoders to keys."""

    encoders: dict[K, Encoder]
    r"""The encoders to map to keys."""

    def __new__[T, X, Y](
        cls, encoders: Mapping[T, Encoder[X, Y]]
    ) -> "MappedEncoder[Mapping[T, X], Mapping[T, Y], T]":
        return super().__new__(cls)  # type: ignore[arg-type]

    def __init__[T, X, Y](
        self: "MappedEncoder[Mapping[T, X], Mapping[T, Y], T]",
        encoders: Mapping[T, Encoder[X, Y]],
    ) -> None:
        super().__init__(encoders)

    def __invert__(self) -> "MappedEncoder[MappingOut, MappingIn, K]":
        # cls: type[MappedEncoder] = type(self)
        decoders = {k: InverseEncoder(e) for k, e in self.encoders.items()}
        return MappedEncoder(decoders)  # type: ignore[return-value]

    def fit(self, xmap: MappingIn, /) -> None:
        if missing_keys := self.encoders.keys() - xmap.keys():
            raise ValueError(f"No data to fit encoders {missing_keys}.")
        if extra_keys := xmap.keys() - self.encoders.keys():
            raise ValueError(f"Extra data with no matching encoder {extra_keys}.")

        for k, x in xmap.items():
            self.encoders[k].fit(x)

    def encode(self, xmap: MappingIn, /) -> MappingOut:
        ymap = {k: self.encoders[k].encode(x) for k, x in xmap.items()}
        return cast(MappingOut, ymap)

    def decode(self, ymap: MappingOut, /) -> MappingIn:
        xmap = {k: self.encoders[k].decode(y) for k, y in ymap.items()}
        return cast(MappingIn, xmap)


def map_encoders[K, X, Y](
    encoders: Mapping[K, Encoder[X, Y]], /, *, simplify: bool = True
) -> IdentityEncoder | MappedEncoder[Mapping[K, X], Mapping[K, Y], K]:
    r"""Map encoders."""
    encoder = MappedEncoder(encoders)
    return encoder.simplify() if simplify else encoder


# @pprint_repr(recursive=2)
# class MapEncoders[
#     MappingIn: Mapping,
#     MappingOut: Mapping,
# ](BaseEncoder[MappingIn, MappingOut], Mapping[Any, Encoder]):
#     r"""Creates an encoder that applies over a mapping."""
#
#     encoders: Mapping[Any, Encoder]
#     r"""Mapping of keys to encoders."""
#
#     @property
#     def params(self) -> dict[str, Any]:
#         return {"encoders": self.encoders}
#
#     @property
#     def requires_fit(self) -> bool:
#         return any(e.requires_fit for e in self.encoders.values())
#
#     @property
#     def is_fitted(self) -> bool:
#         return all(e.is_fitted for e in self.encoders.values())
#
#     @is_fitted.setter
#     def is_fitted(self, value: bool) -> None:
#         for encoder in self.encoders.values():
#             encoder.is_fitted = value
#
#     def __new__[K, X, Y](
#         cls, encoders: Mapping[K, Encoder[X, Y]]
#     ) -> "MapEncoders[Mapping[K, X], Mapping[K, Y]]":
#         return super().__new__(cls)  # type: ignore[arg-type]
#
#     def __init__[K, X, Y](
#         self: "MapEncoders[Mapping[K, X], Mapping[K, Y]]",
#         encoders: Mapping[K, Encoder[X, Y]],
#     ) -> None:
#         self.encoders = encoders
#
#     def __invert__(self) -> "MapEncoders[MappingOut, MappingIn]":
#         decoders = {k: InverseEncoder(e) for k, e in self.encoders.items()}
#         cls: type[MapEncoders] = type(self)
#         return cls(decoders)  # type: ignore[return-value]
#
#     def __getitem__(self, key: Any, /) -> Encoder:
#         r"""Get the encoder for the given key."""
#         return self.encoders[key]
#
#     def __len__(self) -> int:
#         return len(self.encoders)
#
#     def __iter__(self) -> Iterator[Any]:
#         return iter(self.encoders)
#
#     def fit(self, xmap: MappingIn, /) -> None:
#         if missing_keys := self.encoders.keys() - xmap.keys():
#             raise ValueError(f"No data to fit encoders {missing_keys}.")
#         if extra_keys := xmap.keys() - self.encoders.keys():
#             raise ValueError(f"Extra data with no matching encoder {extra_keys}.")
#
#         for k, x in xmap.items():
#             self.encoders[k].fit(x)
#
#     def encode(self, xmap: MappingIn, /) -> MappingOut:
#         ymap = {k: self.encoders[k].encode(x) for k, x in xmap.items()}
#         return cast(MappingOut, ymap)
#
#     def decode(self, ymap: MappingOut, /) -> MappingIn:
#         xmap = {k: self.encoders[k].decode(y) for k, y in ymap.items()}
#         return cast(MappingIn, xmap)
#
#     def simplify(self) -> IdentityEncoder | Self:
#         r"""Simplify the mapping encoder."""
#         # FIXME: https://github.com/python/mypy/issues/17134
#         # match self:
#         #     case []:
#         #         return IdentityEncoder()
#         #     case [encoder]:
#         #         return encoder.simplify()
#         #     case _:
#         #         cls = type(self)
#         #         return cls(*(e.simplify() for e in self))
#
#         if len(self.encoders) == 0:
#             return IdentityEncoder()
#         cls: type[Self] = type(self)
#         return cls({k: e.simplify() for k, e in self.encoders.items()})


# endregion variadic encoders ----------------------------------------------------------
