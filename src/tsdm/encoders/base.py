r"""Base Classes for Encoders."""

__all__ = [
    # ABCs & Protocols
    "BaseEncoder",
    "Encoder",
    "EncoderProtocol",
    "InvertibleTransform",
    "ParametrizedEncoder",
    "SerializableEncoder",
    "Transform",
    # Classes
    "ChainedEncoder",
    "CloneEncoder",
    "DeepcopyEncoder",
    "DiagonalEncoder",
    "DuplicateEncoder",
    "FactorizedEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "MappingEncoder",
    "ParallelEncoder",
    "PipedEncoder",
    "TupleDecoder",
    "TupleEncoder",
    # Functions
    "chain_encoders",
    "parallel_encoders",
    "duplicate_encoder",
    "invert_encoder",
    "pipe_encoders",
    "pow_encoder",
]

import logging
import pickle
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass
from functools import wraps
from inspect import getattr_static

from typing_extensions import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Protocol,
    Self,
    TypeVar,
    deprecated,
    overload,
    runtime_checkable,
)

from tsdm.constants import EMPTY_MAP
from tsdm.types.aliases import FilePath
from tsdm.types.variables import K, T
from tsdm.utils.pprint import pprint_repr

U = TypeVar("U")
U_contra = TypeVar("U_contra", contravariant=True)
V = TypeVar("V")
V_co = TypeVar("V_co", covariant=True)
W = TypeVar("W")
X = TypeVar("X")
Y = TypeVar("Y")


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

    @property
    @abstractmethod
    def is_serializable(self) -> bool: ...
    @abstractmethod
    def serialize(self, filepath: FilePath, /) -> None: ...
    @classmethod
    @abstractmethod
    def deserialize(cls, filepath: FilePath, /) -> Self: ...


class Encoder(Protocol[U, V]):
    r"""Protocol for Encoders with algebraic mixin methods."""

    # region abstract methods ----------------------------------------------------------
    @abstractmethod
    def fit(self, data: U, /) -> None: ...
    @abstractmethod
    def encode(self, data: U, /) -> V: ...
    @abstractmethod
    def decode(self, data: V, /) -> U: ...
    @property
    @abstractmethod
    def requires_fit(self) -> bool: ...

    is_fitted: bool  # NOTE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020
    # endregion abstract methods -------------------------------------------------------

    # region mixin methods -------------------------------------------------------------
    def simplify(self) -> "Encoder[U, V]":
        r"""Simplify the encoder."""
        return self

    # region parameter methods ---------------------------------------------------------
    @property
    def params(self) -> dict[str, Any]: ...

    @property
    def required_params(self) -> frozenset[str]:
        raise NotImplementedError

    def set_params(self, mapping: Mapping[str, Any] = EMPTY_MAP, **kwargs: Any) -> None:
        raise NotImplementedError

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        return self.params

    # endregion parameter methods ------------------------------------------------------

    # region serialization methods -----------------------------------------------------
    def is_serializable(self) -> bool:
        r"""Check if the encoder is serializable."""
        params = self.params
        return not any(params[key] is NotImplemented for key in self.required_params)

    def serialize(self, filepath: FilePath, /) -> None:
        r"""Serialize the encoder to a file."""
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

    # region method aliases ------------------------------------------------------------
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

    # endregion method aliases ---------------------------------------------------------

    # region magic methods -------------------------------------------------------------
    @deprecated("use .encode(data) instead.")
    def __call__(self, data: U, /) -> V:
        r"""Apply the encoder."""
        return self.encode(data)

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
            enc = self @ other
            enc(x) == self(other(x))

        Raises:
            TypeError if other is not an encoder.
        """
        return ChainedEncoder(self, other)

    def __rmatmul__(self, other: "Encoder[V, W]", /) -> "Encoder[U, W]":
        r"""Chain the encoders (pure function composition).

        Example:
            >>> x = ...
            >>> enc = other @ self
            >>> enc(x) == other(self(x))
        """
        return ChainedEncoder(other, self)

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
        return PipedEncoder(self, other)

    def __or__(self, other: "Encoder[X, Y]", /) -> "Encoder[tuple[U, X], tuple[V, Y]]":
        r"""Return product encoders.

        Example:
            enc = self | other
            enc((x, y)) == (self(x), other(y))
        """
        return ParallelEncoder(self, other)

    def __ror__(self, other: "Encoder[X, Y]", /) -> "Encoder[tuple[X, U], tuple[Y, V]]":
        r"""Return product encoders.

        Example:
            enc = other | self
            enc((x, y)) == (other(x), self(y))
        """
        return ParallelEncoder(other, self)

    # endregion magic methods ----------------------------------------------------------
    # endregion mixin methods ----------------------------------------------------------


E = TypeVar("E", bound=Encoder)
r"""Type alias for Encoder."""


class BaseEncoder(Encoder[U, V]):
    r"""Base class for encoders implemented within this package."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the Encoder."""

    _is_fitted: bool = False
    r"""Whether the encoder has been fitted."""

    # region abstract methods ----------------------------------------------------------
    @property
    @abstractmethod
    def params(self) -> dict[str, Any]: ...

    @property
    def requires_fit(self) -> bool:
        r"""Check if the encoder requires fitting."""
        return any(
            val is NotImplemented or getattr(val, "requires_fit", False)
            for val in self.params.values()
        )

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
            if self.requires_fit:
                self.LOGGER.info("Fitting encoder to data.")
                original_fit(self, data)
            else:
                self.LOGGER.info(
                    "Skipping fitting as encoder does not require fitting."
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

    @property
    def is_fitted(self) -> bool:
        r"""Whether the encoder has been fitted."""
        return (not self.requires_fit) or self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool, /) -> None:
        self._is_fitted = value

    @property
    def required_params(self) -> frozenset[str]:
        r"""The required parameters of the encoder."""
        return frozenset(self.params.keys())

    def is_serializable(self) -> bool:
        r"""Check if the encoder is serializable.

        By default, an encoder can be serialized if all of its required parameters are initialized.
        """
        params = self.params
        return any(params[k] is NotImplemented for k in self.required_params)

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        r"""Return the parameters of the encoder.

        Compatibility interface with scikit-learn.
        Use the property `self.params` to access the parameters directly.

        Uninitialized required parameters are marked as `NotImplemented`.
        Uninitialized optional parameters are marked as `None`.
        """
        return self.params

    def set_params(
        self, mapping: Mapping[str, Any] = EMPTY_MAP, /, **kwargs: Any
    ) -> None:
        r"""Set the parameters of the encoder."""
        self.__dict__.update(mapping, **kwargs)

    # region chaining methods ----------------------------------------------------------
    # def pow(self, power: int, /) -> "PipedEncoder":
    #     r"""Return the chain of itself multiple times."""
    #     return pow_encoder(self, power)
    #
    # def standardize(self) -> "ChainedEncoder[Self, StandardScaler]":
    #     r"""Chain a standardizer."""
    #     return self >> StandardScaler()
    #
    # def minmax_scale(self) -> "ChainedEncoder[Self, MinMaxScaler]":
    #     r"""Chain a minmax scaling."""
    #     return self >> MinMaxScaler()
    #
    # def normalize(self) -> "ChainedEncoder[Self, Normalizer]":
    #     r"""Chain a normalization."""
    #     return self >> Normalizer()
    #
    # def quantile_transform(self) -> "ChainedEncoder[Self, QuantileTransformer]":
    #     r"""Chain a quantile transformation."""
    #     return self >> QuantileTransformer()
    # endregion chaining methods -------------------------------------------------------


class IdentityEncoder(BaseEncoder[Any, Any]):
    r"""Dummy class that performs identity function."""

    requires_fit: ClassVar[bool] = False

    def encode(self, data: T, /) -> T:
        return data

    def decode(self, data: T, /) -> T:
        return data


@dataclass
class DiagonalEncoder(BaseEncoder[T, tuple[T, ...]]):
    r"""Encodes the input into a tuple of itself.

    .. math:: f(x) = (x, x, …, x)

    Note:
        In practice, when working with float arrays, we need to be careful how to select
        the inverse. Due to rounding errors, the values in the tuple elements might be
        slightly different. In this case, an aggregation function needs to be supplied.
    """

    requires_fit: ClassVar[bool] = False

    num: int

    _: KW_ONLY

    aggregate_fn: Optional[Callable[[tuple[T, ...]], T]] = None

    def encode(self, data: T, /) -> tuple[T, ...]:
        return (data,) * self.num

    def decode(self, data: tuple[T, ...], /) -> T:
        if self.aggregate_fn is None:
            try:
                vals = set(data)
            except TypeError as exc:
                raise TypeError(
                    "Data not hashable, please provide an aggregate_fn."
                ) from exc
            if len(vals) != 1:
                raise ValueError("Data not constant, please provide an aggregate_fn.")
            return vals.pop()

        return self.aggregate_fn(data)


class TupleEncoder(BaseEncoder):
    r"""Wraps input into a tuple."""

    requires_fit: ClassVar[bool] = False

    def __invert__(self) -> "TupleDecoder":
        return TupleDecoder()

    def encode(self, data: T, /) -> tuple[T]:
        return (data,)

    def decode(self, data: tuple[T], /) -> T:
        return data[0]


class TupleDecoder(BaseEncoder):
    r"""Unwraps input from a tuple."""

    requires_fit: ClassVar[bool] = False

    def __invert__(self) -> "TupleEncoder":
        return TupleEncoder()

    def encode(self, data: tuple[T], /) -> T:
        return data[0]

    def decode(self, data: T, /) -> tuple[T]:
        return (data,)


class InverseEncoder(BaseEncoder[U, V]):
    r"""Applies an encoder in reverse."""

    encoder: Encoder[V, U]
    r"""The encoder to invert."""

    def __init__(self, encoder: Encoder[V, U], /) -> None:
        self.encoder = encoder

    def __repr__(self) -> str:
        return f"~{self.encoder}"

    def fit(self, data: T, /) -> None:
        raise NotImplementedError("Inverse encoders cannot be fitted.")

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def encode(self, data: U, /) -> V:
        return self.encoder.decode(data)

    def decode(self, data: V, /) -> U:
        return self.encoder.encode(data)

    def simplify(self) -> Self:
        cls = type(self)
        return cls(self.encoder.simplify())


def invert_encoder(encoder: Encoder[U, V], /) -> Encoder[V, U]:
    r"""Return the inverse encoder (i.e. decoder)."""
    return ~encoder


@pprint_repr(recursive=2)
class ChainedEncoder(BaseEncoder, Sequence[Encoder]):
    r"""Represents function composition of encoders."""

    encoders: list[Encoder]
    r"""List of encoders."""

    def __init__(self, *encoders: Encoder) -> None:
        self.encoders = list(encoders)

    def __invert__(self) -> Self:
        cls = type(self)
        return cls(*(~e for e in reversed(self.encoders)))

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
                return ChainedEncoder(*self.encoders[slc])
            case _:
                raise TypeError(f"Type {type(index)} not supported.")

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

    def fit(self, data: Any, /) -> None:
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

    def encode(self, data: Any, /) -> Any:
        for encoder in reversed(self.encoders):
            data = encoder.encode(data)
        return data

    def decode(self, data: Any, /) -> Any:
        for encoder in self.encoders:
            data = encoder.decode(data)
        return data

    def simplify(self) -> IdentityEncoder | Encoder | Self:
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
                return type(self)(*encoders)


@overload
def chain_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
@overload
def chain_encoders(e: Encoder, /, *, simplify: Literal[True]) -> Encoder: ...
@overload
def chain_encoders(*encoders: Encoder, simplify: bool = ...) -> ChainedEncoder: ...
def chain_encoders(*encoders, simplify=True):
    r"""Chain encoders."""
    encoder = ChainedEncoder(*encoders)
    return encoder.simplify() if simplify else encoder


@pprint_repr(recursive=2)
class PipedEncoder(BaseEncoder, Sequence[Encoder]):
    r"""Represents function composition of encoders."""

    encoders: list[Encoder]
    r"""List of encoders."""

    def __init__(self, *encoders: Encoder) -> None:
        self.encoders = list(encoders)

    def __invert__(self) -> Self:
        cls = type(self)
        return cls(*(~e for e in reversed(self.encoders)))

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
                return PipedEncoder(*self.encoders[slc])
            case _:
                raise TypeError(f"Type {type(index)} not supported.")

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

    def fit(self, data: Any, /) -> None:
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

    def encode(self, data: Any, /) -> Any:
        for encoder in self.encoders:
            data = encoder.encode(data)
        return data

    def decode(self, data: Any, /) -> Any:
        for encoder in reversed(self.encoders):
            data = encoder.decode(data)
        return data

    def simplify(self) -> IdentityEncoder | Encoder | Self:
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
                return type(self)(*encoders)


@overload
def pipe_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
@overload
def pipe_encoders(e: Encoder, /, *, simplify: Literal[True]) -> Encoder: ...
@overload
def pipe_encoders(*encoders: Encoder, simplify: bool = ...) -> PipedEncoder: ...
def pipe_encoders(*encoders, simplify=True):
    r"""Chain encoders."""
    encoder = PipedEncoder(*encoders)
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
    return PipedEncoder(*encoders)


@pprint_repr(recursive=2)
class ParallelEncoder(BaseEncoder[tuple[Any, ...], tuple[Any, ...]], Sequence[Encoder]):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """

    encoders: list[Encoder]
    r"""The encoders."""

    def __init__(self, *encoders: Encoder) -> None:
        self.encoders = list(encoders)

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
        r"""Return the number of the encoders."""
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
                return ParallelEncoder(*self.encoders[slc])
            case _:
                raise TypeError(f"Type {type(index)} not supported.")

    def fit(self, data: tuple[Any, ...], /) -> None:
        for encoder, x in zip(self.encoders, data, strict=True):
            encoder.fit(x)

    def encode(self, data: tuple[Any, ...], /) -> tuple[Any, ...]:
        rtype = type(data)
        return rtype(
            encoder.encode(x) for encoder, x in zip(self.encoders, data, strict=True)
        )

    def decode(self, data: tuple[Any, ...], /) -> tuple[Any, ...]:
        rtype = type(data)
        return rtype(
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
        return cls(*(e.simplify() for e in self.encoders))


@overload
def parallel_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
@overload
def parallel_encoders(e: Encoder, /, *, simplify: Literal[True]) -> Encoder: ...
@overload
def parallel_encoders(
    e1: Encoder, e2: Encoder, /, *encoders: Encoder, simplify: bool = ...
) -> ParallelEncoder: ...
def parallel_encoders(*encoders, simplify=True):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """
    encoder = ParallelEncoder(*encoders)
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
        return parallel_encoders(*[~e for e in reversed(encoders)])
    return parallel_encoders(*encoders)


@pprint_repr(recursive=2)
class FactorizedEncoder(BaseEncoder[U, tuple[V, ...]], Sequence[Encoder]):
    r"""Factorized Encoder.

    Example:
        enc = FactorizedEncoder(e1, e2, e3)
        enc(x) == (e1(x), e2(x), e3(x))

    Note:
        This is essentially equivalent to chaining `DiagonalEncoder >> ParallelEncoder`.
    """

    encoders: list[Encoder[U, V]]
    aggregate_fn: Optional[Callable[[list[U]], U]] = None

    def __init__(
        self,
        *encoders: Encoder[U, V],
        aggregate_fn: Optional[Callable[[list[U]], U]] = None,
    ) -> None:
        self.encoders = list(encoders)
        self.aggregate_fn = aggregate_fn

    def __len__(self) -> int:
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
                return FactorizedEncoder(*self.encoders[slc])
            case _:
                raise TypeError(f"Type {type(index)} not supported.")

    def encode(self, data: U, /) -> tuple[V, ...]:
        return tuple(e.encode(data) for e in self.encoders)

    def decode(self, data: tuple[V, ...], /) -> U:
        decoded_vals = [e.decode(x) for e, x in zip(self.encoders, data, strict=True)]

        if self.aggregate_fn is None:
            try:
                vals = set(decoded_vals)
            except TypeError as exc:
                raise TypeError(
                    "Data not hashable, please provide an aggregate_fn."
                ) from exc
            if len(vals) != 1:
                raise ValueError("Data not constant, please provide an aggregate_fn.")
            return vals.pop()
        return self.aggregate_fn(decoded_vals)


@pprint_repr(recursive=2)
class MappingEncoder(
    BaseEncoder[Mapping[K, Any], Mapping[K, Any]],
    Mapping[K, Encoder],
):
    r"""Creates an encoder that applies over a mapping."""

    encoders: Mapping[K, Encoder]
    r"""Mapping of keys to encoders."""

    @property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self.encoders.values())

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

    def simplify(self) -> IdentityEncoder | Self:
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


class DeepcopyEncoder(BaseEncoder):
    r"""Encoder that deepcopies the input."""

    requires_fit: ClassVar[bool] = False

    def encode(self, data: T, /) -> T:
        return deepcopy(data)

    def decode(self, data: T, /) -> T:
        return deepcopy(data)


class DuplicateEncoder(BaseEncoder[tuple[U, ...], tuple[V, ...]]):
    r"""Duplicate encoder multiple times (references same object)."""

    base_encoder: Encoder[U, V]

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def __init__(self, encoder: Encoder[U, V], n: int = 1) -> None:
        self.base_encoder = encoder
        self.n = n
        self.encoder = ParallelEncoder(*(self.base_encoder for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: tuple[U, ...], /) -> None:
        return self.encoder.fit(data)

    def encode(self, data: tuple[U, ...], /) -> tuple[V, ...]:
        return self.encoder.encode(data)

    def decode(self, data: tuple[V, ...], /) -> tuple[U, ...]:
        return self.encoder.decode(data)


class CloneEncoder(BaseEncoder[tuple[U, ...], tuple[V, ...]]):
    r"""Clone encoder multiple times (distinct copies)."""

    base_encoder: Encoder[U, V]
    n: int
    encoder: ParallelEncoder

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def __init__(self, encoder: Encoder[U, V], n: int = 1) -> None:
        self.base_encoder = encoder
        self.n = n
        self.encoder = ParallelEncoder(*(deepcopy(self.base_encoder) for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: tuple[U, ...], /) -> None:
        return self.encoder.fit(data)

    def encode(self, data: tuple[U, ...], /) -> tuple[V, ...]:
        return self.encoder.encode(data)

    def decode(self, data: tuple[V, ...], /) -> tuple[U, ...]:
        return self.encoder.decode(data)
