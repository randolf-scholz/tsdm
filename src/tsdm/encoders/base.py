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

__all__ = [
    # ABCs & Protocols
    "BaseEncoder",
    "Encoder",
    "EncoderList",
    "EncoderProtocol",
    "InvertibleTransform",
    "ParametrizedEncoder",
    "SerializableEncoder",
    "Transform",
    # Classes
    "ChainedEncoder",
    "DeepcopyEncoder",
    "DiagonalEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "JointEncoder",
    "MappedEncoders",
    "ParallelEncoder",
    "PipedEncoder",
    "TupleDecoder",
    "TupleEncoder",
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
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import KW_ONLY, asdict, dataclass
from functools import wraps
from inspect import getattr_static

from typing_extensions import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypeVar,
    TypeVarTuple,
    cast,
    overload,
    runtime_checkable,
)

from tsdm import encoders as E
from tsdm.types.aliases import FilePath
from tsdm.types.variables import K, T
from tsdm.utils.decorators import pprint_repr

U = TypeVar("U")
U_contra = TypeVar("U_contra", contravariant=True)
V = TypeVar("V")
V_co = TypeVar("V_co", covariant=True)
W = TypeVar("W")
X = TypeVar("X")
Y = TypeVar("Y")
Xs = TypeVarTuple("Xs")
Ys = TypeVarTuple("Ys")


@runtime_checkable
class Transform(Protocol[U_contra, V_co]):
    r"""Protocol for transformers."""

    @abstractmethod
    def fit(self, data: U_contra, /) -> None: ...
    @abstractmethod
    def transform(self, data: U_contra, /) -> V_co: ...


@runtime_checkable
class InvertibleTransform(Transform[U, V], Protocol):
    r"""Protocol for invertible transformers."""

    @abstractmethod
    def inverse_transform(self, data: V, /) -> U: ...


@runtime_checkable
class EncoderProtocol(Protocol[U, V]):
    r"""Minimal Protocol for Encoders.

    This protocol should be used in applications that only use encoders, but do not need to
    worry about creating new encoders or chaining them together.
    """

    @abstractmethod
    def fit(self, data: U, /) -> None: ...
    @abstractmethod
    def encode(self, data: U, /) -> V: ...
    @abstractmethod
    def decode(self, data: V, /) -> U: ...


class ParametrizedEncoder(EncoderProtocol[U, V], Protocol):
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

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        r"""Alias for `self.params`."""
        return self.params

    # endregion mixin methods ---------------------------------------------------------


class SerializableEncoder(EncoderProtocol[U, V], Protocol):
    r"""Protocol for serializable encoders."""

    @classmethod
    @abstractmethod
    def deserialize(cls, filepath: FilePath, /) -> Self: ...
    @abstractmethod
    def serialize(self, filepath: FilePath, /) -> None: ...


class Encoder(Protocol[U, V]):
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
    def fit(self, data: U, /) -> None: ...
    @abstractmethod
    def encode(self, data: U, /) -> V: ...
    @abstractmethod
    def decode(self, data: V, /) -> U: ...

    # endregion abstract methods -------------------------------------------------------

    @property
    def required_params(self) -> frozenset[str]:
        r"""The required parameters to initialize the encoder."""
        return frozenset(self.params)

    def simplify(self) -> "Encoder[U, V]":
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
            assert isinstance(obj, cls)
            return obj

    # endregion serialization methods --------------------------------------------------

    # region scikit-learn compatibility ------------------------------------------------
    def fit_transform(self, data: U, /) -> V:
        r"""Fit the encoder to the data and apply the transformation."""
        self.fit(data)
        return self.encode(data)

    def transform(self, data: U, /) -> V:
        r"""Alias for encode."""
        return self.encode(data)

    def inverse_transform(self, data: V, /) -> U:
        r"""Alias for decode."""
        return self.decode(data)

    def set_params(self, **kwargs: Any) -> None:
        r"""Compatibility interface with scikit-learn."""
        try:
            self.__dict__.update(**kwargs)
        except Exception as exc:
            exc.add_note(f"failed to set parameters {kwargs}.")
            raise

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        """Compatibility interface with scikit-learn.

        Compatibility interface with scikit-learn.
        Use the property `self.params` to access the parameters directly.

        Uninitialized required parameters should be marked as `NotImplemented`.
        Uninitialized optional parameters should be marked as `None`.
        """
        return self.params

    # endregion scikit-learn compatibility ---------------------------------------------

    # region magic methods -------------------------------------------------------------
    def __invert__(self) -> "Encoder[V, U]":
        r"""Return the inverse encoder (i.e. decoder).

        Example:
            enc = ~self
            enc(y) == self.decode(y)
        """
        return InverseEncoder(self)

    def __matmul__(self, other: "Encoder[T, U]", /) -> "Encoder[T, V]":
        r"""Chain the encoders (pure function composition).

        Example:
            >>> x = ...
            >>> enc = other @ self
            >>> enc.encode(x) == other.encode(self.encode(x))

        Raises:
            TypeError if other is not an encoder.
        """
        return ChainedEncoder((self, other))

    def __rmatmul__(self, other: "Encoder[V, W]", /) -> "Encoder[U, W]":
        r"""Chain the encoders (pure function composition).

        See `__matmul__` for more details.
        """
        return ChainedEncoder((other, self))

    def __rshift__(self, other: "Encoder[V, W]", /) -> "Encoder[U, W]":
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
        return PipedEncoder((self, other))

    def __rrshift__(self, other: "Encoder[T, U]", /) -> "Encoder[T, V]":
        r"""Pipe the encoders (encoder composition).

        See `__rshift__` for more details.
        """
        return PipedEncoder((other, self))

    def __or__(self, other: "Encoder[X, Y]", /) -> "Encoder[tuple[U, X], tuple[V, Y]]":
        r"""Return product encoders.

        Example:
            enc = self | other
            enc((x, y)) == (self(x), other(y))
        """
        return ParallelEncoder((self, other))

    def __ror__(self, other: "Encoder[X, Y]", /) -> "Encoder[tuple[X, U], tuple[Y, V]]":
        r"""Return product encoders.

        See `__or__` for more details.
        """
        return ParallelEncoder((other, self))

    # endregion magic methods ----------------------------------------------------------


class BaseEncoder(Encoder[U, V]):
    r"""Base class for encoders implemented within this package."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the Encoder."""

    _is_fitted: bool = False
    r"""Whether the encoder has been fitted."""

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
    @property
    @abstractmethod
    def params(self) -> dict[str, Any]: ...

    @abstractmethod
    def encode(self, data: U, /) -> V:
        r"""Encode the data by transformation."""
        ...

    @abstractmethod
    def decode(self, data: V, /) -> U:
        r"""Decode the data by inverse transformation."""
        ...

    def fit(self, data: U, /) -> None:
        r"""Implement as necessary."""

    def simplify(self) -> Encoder[U, V]:
        r"""Simplify the encoder."""
        return self

    # endregion abstract methods -------------------------------------------------------

    def __init_subclass__(cls) -> None:
        r"""Initialize the subclass.

        The wrapping of fit/encode/decode must be done here to avoid `~pickle.PickleError`!
        """
        super().__init_subclass__()  # <-- This is important! Otherwise, weird things happen.

        for meth in ("fit", "encode", "decode"):
            static_meth = getattr_static(cls, meth, None)
            if static_meth is None:
                raise NotImplementedError(f"Missing method {meth}.")
            if isinstance(static_meth, staticmethod | classmethod):
                raise TypeError(f"Method {meth} can't be static/class method.")

        original_fit = cls.fit
        original_encode = cls.encode
        original_decode = cls.decode

        @wraps(original_fit)
        def fit_wrapper(self: Self, data: U, /) -> None:
            r"""Fit the encoder to the data."""
            self.LOGGER.info("Fitting encoder to data.")
            original_fit(self, data)

            # check if fitting was successful
            if self.requires_fit:
                raise AssertionError(
                    "Fitting was not successful! "
                    "Possibly the encoder is not implemented correctly."
                )
            self.is_fitted = True

        @wraps(original_encode)
        def encode_wrapper(self: Self, data: U, /) -> V:
            r"""Encode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted!")
            return original_encode(self, data)

        @wraps(original_decode)
        def decode_wrapper(self: Self, data: V, /) -> U:
            r"""Decode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted!")
            return original_decode(self, data)

        cls.fit = fit_wrapper  # type: ignore[assignment]
        cls.encode = encode_wrapper  # type: ignore[assignment]
        cls.decode = decode_wrapper  # type: ignore[assignment]

    # region magic methods -------------------------------------------------------------
    def __invert__(self) -> "BaseEncoder[V, U]":
        r"""Return the inverse encoder (i.e. decoder).

        Example:
            enc = ~self
            enc(y) == self.decode(y)
        """
        return InverseEncoder(self)

    def __matmul__(self, other: "Encoder[T, U]", /) -> "ChainedEncoder[T, V]":
        r"""Chain the encoders (pure function composition).

        Example:
            >>> x = ...
            >>> enc = other @ self
            >>> enc.encode(x) == other.encode(self.encode(x))

        Raises:
            TypeError if other is not an encoder.
        """
        return ChainedEncoder((self, other))

    def __rmatmul__(self, other: "Encoder[V, W]", /) -> "ChainedEncoder[U, W]":
        r"""Chain the encoders (pure function composition).

        See `__matmul__` for more details.
        """
        return ChainedEncoder((other, self))

    def __rshift__(self, other: "Encoder[V, W]", /) -> "PipedEncoder[U, W]":
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
        return PipedEncoder((self, other))

    def __rrshift__(self, other: "Encoder[T, U]", /) -> "PipedEncoder[T, V]":
        r"""Pipe the encoders (encoder composition).

        See `__rshift__` for more details.
        """
        return PipedEncoder((other, self))

    def __or__(  # type: ignore[override]
        self, other: "Encoder[X, Y]", /
    ) -> "ParallelEncoder[tuple[U, X], tuple[V, Y]]":
        r"""Return product encoders.

        Example:
            enc = self | other
            enc((x, y)) == (self(x), other(y))
        """
        return ParallelEncoder((self, other))

    def __ror__(  # type: ignore[override]
        self, other: "Encoder[X, Y]", /
    ) -> "ParallelEncoder[tuple[X, U], tuple[Y, V]]":
        r"""Return product encoders.

        See `__or__` for more details.
        """
        return ParallelEncoder((other, self))

    # endregion magic methods ----------------------------------------------------------

    # region chaining methods ----------------------------------------------------------
    def standardize(self) -> "BaseEncoder[U, V]":
        r"""Chain a standardizer."""
        return self >> E.StandardScaler()

    def minmax_scale(self) -> "BaseEncoder[U, V]":
        r"""Chain a minmax scaling."""
        return self >> E.MinMaxScaler()

    # endregion chaining methods -------------------------------------------------------


# region elementary encoders -----------------------------------------------------------
class IdentityEncoder(BaseEncoder[Any, Any]):
    r"""Dummy class that performs identity function."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def encode(self, data: T, /) -> T:
        return data

    def decode(self, data: T, /) -> T:
        return data


class DeepcopyEncoder(BaseEncoder):
    r"""Encoder that deepcopies the input."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def encode(self, data: T, /) -> T:
        return deepcopy(data)

    def decode(self, data: T, /) -> T:
        return deepcopy(data)


class TupleEncoder(BaseEncoder):
    r"""Wraps input into a tuple."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def __invert__(self) -> "TupleDecoder":
        return TupleDecoder()

    def encode(self, data: T, /) -> tuple[T]:
        return (data,)

    def decode(self, data: tuple[T], /) -> T:
        return data[0]


class TupleDecoder(BaseEncoder):
    r"""Unwraps input from a tuple."""

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def __invert__(self) -> "TupleEncoder":
        return TupleEncoder()

    def encode(self, data: tuple[T], /) -> T:
        return data[0]

    def decode(self, data: T, /) -> tuple[T]:
        return (data,)


@dataclass
class DiagonalEncoder(BaseEncoder[T, tuple[T, ...]]):
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

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def encode(self, data: T, /) -> tuple[T, ...]:
        return (data,) * self.num

    def decode(self, data: tuple[T, ...], /) -> T:
        return self.aggregate_fn(data)


# endregion elementary encoders --------------------------------------------------------


@dataclass(frozen=True, slots=True, repr=False)
class InverseEncoder(BaseEncoder[U, V]):
    r"""Applies an encoder in reverse."""

    encoder: Encoder[V, U]
    r"""The encoder to invert."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def fit(self, data: U, /) -> None:
        raise NotImplementedError("Inverse encoders cannot be fitted.")

    def encode(self, data: U, /) -> V:
        return self.encoder.decode(data)

    def decode(self, data: V, /) -> U:
        return self.encoder.encode(data)

    def simplify(self) -> Self:
        cls = type(self)
        return cls(self.encoder.simplify())

    def __repr__(self) -> str:
        return f"~{self.encoder}"


def invert_encoder(encoder: Encoder[U, V], /) -> Encoder[V, U]:
    r"""Return the inverse encoder (i.e. decoder)."""
    return ~encoder


class EncoderList(BaseEncoder[U, V], Sequence[Encoder]):
    r"""List of encoders."""

    encoders: list[Encoder]

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

    def __init__(self, encoders: Iterable[Encoder], /) -> None:
        self.encoders = list(encoders)

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
                return cls(self.encoders[slc])
            case _:
                raise TypeError(f"Type {type(index)} not supported.")


@pprint_repr(recursive=2)
class ChainedEncoder(EncoderList[U, V]):
    r"""Represents function composition of encoders."""

    def __invert__(self) -> "ChainedEncoder[V, U]":
        cls: type[ChainedEncoder] = type(self)
        return cls(~e for e in reversed(self.encoders))

    def fit(self, data: U, /) -> None:
        for encoder in reversed(self.encoders):
            try:
                encoder.fit(data)
            except Exception as exc:
                index = self.encoders.index(encoder)
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{index}]: Failed to fit {enc!r}.")
                raise
            data = encoder.encode(data)

    def encode(self, data: U, /) -> V:
        for encoder in reversed(self.encoders):
            data = encoder.encode(data)
        return cast(V, data)

    def decode(self, data: V, /) -> U:
        for encoder in self.encoders:
            data = encoder.decode(data)
        return cast(U, data)

    def simplify(self) -> IdentityEncoder | Encoder[U, V] | Self:
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
                return type(self)(encoders)


# fmt: off
@overload
def chain_encoders(*, simplify: Literal[True] = ...) -> IdentityEncoder: ...
@overload
def chain_encoders(e: Encoder[U, V], /, *, simplify: Literal[True] = ...) -> Encoder[U, V]: ...
@overload
def chain_encoders(*es: Encoder, simplify: Literal[False] = ...) -> ChainedEncoder: ...
# fmt: on
def chain_encoders(*encoders: Encoder, simplify: bool = True) -> Encoder:
    r"""Chain encoders."""
    encoder = ChainedEncoder(encoders)
    return encoder.simplify() if simplify else encoder


@pprint_repr(recursive=2)
class PipedEncoder(EncoderList[U, V]):
    r"""Represents function composition of encoders."""

    def __invert__(self) -> "PipedEncoder[V, U]":
        cls: type[PipedEncoder] = type(self)
        return cls(~e for e in reversed(self.encoders))

    def fit(self, data: U, /) -> None:
        for encoder in self.encoders:
            try:
                encoder.fit(data)
            except Exception as exc:
                index = self.encoders.index(encoder)
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{index}]: Failed to fit {enc!r}.")
                raise
            data = encoder.encode(data)

    def encode(self, data: U, /) -> V:
        for encoder in self.encoders:
            data = encoder.encode(data)
        return cast(V, data)

    def decode(self, data: V, /) -> U:
        for encoder in reversed(self.encoders):
            data = encoder.decode(data)
        return cast(U, data)

    def simplify(self) -> IdentityEncoder | Encoder[U, V] | Self:
        r"""Simplify the chained encoder."""
        # simplify the nested encoders
        encoders = []
        for encoder in (e.simplify() for e in self):
            match encoder:
                case PipedEncoder(encoders=nested):
                    encoders.extend(nested)
                case ChainedEncoder(encoders=nested):
                    encoders.extend(reversed(nested))
                case _:
                    encoders.append(encoder)

        # simplify self
        match encoders:
            case []:
                return IdentityEncoder()
            case [encoder]:
                return encoder
            case _:
                return type(self)(encoders)


# fmt: off
@overload
def pipe_encoders(*, simplify: Literal[True] = ...) -> IdentityEncoder: ...
@overload
def pipe_encoders(e: Encoder[U, V], /, *, simplify: Literal[True] = ...) -> Encoder[U, V]: ...
@overload
def pipe_encoders(*es: Encoder, simplify: Literal[False] = ...) -> PipedEncoder: ...
# fmt: on
def pipe_encoders(*encoders: Encoder, simplify: bool = True) -> Encoder:
    r"""Pipe encoders."""
    encoder = PipedEncoder(encoders)
    return encoder.simplify() if simplify else encoder


@overload
def pow_encoder(
    e: Encoder,
    n: Literal[0],
    /,
    *,
    simplify: Literal[True],
    copy: bool = ...,
) -> IdentityEncoder: ...
@overload
def pow_encoder(
    e: Encoder, n: Literal[1], /, *, simplify: Literal[True], copy: bool = ...
) -> Encoder: ...
@overload
def pow_encoder(
    e: Encoder, n: int, /, *, simplify: bool = ..., copy: bool = ...
) -> PipedEncoder: ...
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
    return PipedEncoder(encoders)


@pprint_repr(recursive=2)
class ParallelEncoder(EncoderList[tuple[U, ...], tuple[V, ...]]):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """

    def fit(self, data: tuple[U, ...], /) -> None:
        for encoder, x in zip(self.encoders, data, strict=True):
            encoder.fit(x)

    def encode(self, data: tuple[U, ...], /) -> tuple[V, ...]:
        return tuple(
            encoder.encode(x) for encoder, x in zip(self.encoders, data, strict=True)
        )

    def decode(self, data: tuple[V, ...], /) -> tuple[U, ...]:
        return tuple(
            encoder.decode(x) for encoder, x in zip(self.encoders, data, strict=True)
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
        return cls(e.simplify() for e in self.encoders)


@overload
def parallelize_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
@overload
def parallelize_encoders(e: Encoder, /, *, simplify: Literal[True]) -> Encoder: ...
@overload
def parallelize_encoders(
    e1: Encoder, e2: Encoder, /, *encoders: Encoder, simplify: bool = ...
) -> ParallelEncoder: ...
def parallelize_encoders(*encoders, simplify=True):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """
    encoder = ParallelEncoder(encoders)
    return encoder.simplify() if simplify else encoder


@overload
def duplicate_encoder(
    e: Encoder,
    n: Literal[0],
    /,
    *,
    simplify: Literal[True],
    copy: bool = ...,
) -> IdentityEncoder: ...
@overload
def duplicate_encoder(
    e: Encoder, n: Literal[1], /, *, simplify: Literal[True], copy: bool = ...
) -> Encoder: ...
@overload
def duplicate_encoder(
    e: Encoder, n: int, /, *, simplify: bool = ..., copy: bool = ...
) -> ParallelEncoder: ...
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
class JointEncoder(EncoderList[U, tuple[V, ...]], Sequence[Encoder]):
    r"""Factorized Encoder.

    Example:
        enc = FactorizedEncoder(e1, e2, e3)
        enc(x) == (e1(x), e2(x), e3(x))

    Note:
        This is essentially equivalent to chaining `DiagonalEncoder >> ParallelEncoder`.
    """

    encoders: list[Encoder[U, V]]
    aggregate_fn: Callable[[list[U]], U] = random.choice

    @property
    def params(self):
        return {"encoders": self.encoders, "aggregate_fn": self.aggregate_fn}

    def __init__(
        self,
        encoders: Iterable[Encoder[U, V]],
        /,
        *,
        aggregate_fn: Callable[[list[U]], U] = random.choice,
    ) -> None:
        super().__init__(encoders)
        self.aggregate_fn = aggregate_fn

    def encode(self, data: U, /) -> tuple[V, ...]:
        return tuple(e.encode(data) for e in self.encoders)

    def decode(self, data: tuple[V, ...], /) -> U:
        decoded_vals = [e.decode(x) for e, x in zip(self.encoders, data, strict=True)]
        return self.aggregate_fn(decoded_vals)

    def simplify(self) -> IdentityEncoder | Encoder[U, tuple[V, ...]] | Self:
        r"""Simplify the joint encoder."""
        # FIXME: https://github.com/python/mypy/issues/17134
        if len(self) == 0:
            return IdentityEncoder()
        if len(self) == 1:
            return (self[0] >> TupleEncoder()).simplify()
        cls = type(self)
        return cls(e.simplify() for e in self)


# fmt: off
@overload
def join_encoders(*, simplify: Literal[True] = ...) -> IdentityEncoder: ...
@overload
def join_encoders(e: Encoder[U, V], /, *, simplify: Literal[True] = ...) -> Encoder[U, tuple[V, ...]]: ...
@overload
def join_encoders(*es: Encoder[U, V], simplify: Literal[False] = ...) -> JointEncoder[U, V]: ...
# fmt: on
def join_encoders(
    *encoders: Encoder[U, V],
    aggregate_fn: Callable[[list[U]], U] = random.choice,
    simplify: bool = True,
) -> Encoder[U, tuple[V, ...]]:
    r"""Join encoders."""
    enc = JointEncoder(encoders, aggregate_fn=aggregate_fn)
    return enc.simplify() if simplify else enc


@pprint_repr(recursive=2)
class MappedEncoders(
    BaseEncoder[Mapping[K, Any], Mapping[K, Any]],
    Mapping[K, Encoder],
):
    r"""Creates an encoder that applies over a mapping."""

    encoders: Mapping[K, Encoder]
    r"""Mapping of keys to encoders."""

    @property
    def params(self) -> dict[str, Any]:
        return {"encoders": self.encoders}

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

    def __init__(self, encoders: Mapping[K, Encoder], /) -> None:
        self.encoders = encoders

    def __getitem__(self, key: K, /) -> Encoder:
        r"""Get the encoder for the given key."""
        return self.encoders[key]

    def __len__(self) -> int:
        return len(self.encoders)

    def __iter__(self) -> Iterator[K]:
        return iter(self.encoders)

    def fit(self, data: Mapping[K, Any], /) -> None:
        assert set(data.keys()) == set(self.encoders.keys())
        for key in data:
            self.encoders[key].fit(data[key])

    def encode(self, data: Mapping[K, Any], /) -> Mapping[K, Any]:
        return {
            key: encoder.encode(data[key]) for key, encoder in self.encoders.items()
        }

    def decode(self, data: Mapping[K, Any], /) -> Mapping[K, Any]:
        return {
            key: encoder.decode(data[key]) for key, encoder in self.encoders.items()
        }

    def simplify(self) -> IdentityEncoder | Encoder | Self:
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
        if len(self.encoders) == 1:
            return next(iter(self.encoders.values())).simplify()
        cls = type(self)
        return cls({k: e.simplify() for k, e in self.encoders.items()})


def map_encoders(
    encoders: Mapping[K, Encoder], /, *, simplify: bool = True
) -> IdentityEncoder | Encoder | MappedEncoders[K]:
    r"""Map encoders."""
    encoder = MappedEncoders(encoders)
    return encoder.simplify() if simplify else encoder
