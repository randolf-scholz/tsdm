r"""Base Classes for Encoders."""

__all__ = [
    # ABCs & Protocols
    "BaseEncoder",
    "Encoder",
    "EncoderProtocol",
    "InvertibleTransform",
    "Transform",
    # Classes
    "ChainedEncoder",
    "CloneEncoder",
    "CopyEncoder",
    "DuplicateEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "MappingEncoder",
    "PipedEncoder",
    "ParallelEncoder",
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
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from functools import wraps
from inspect import getattr_static

from typing_extensions import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypeVar,
    overload,
    runtime_checkable,
)

from tsdm.types.variables import T2, K, T
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
    """Protocol for transformers."""

    @abstractmethod
    def fit(self, data: U_contra, /) -> None: ...
    @abstractmethod
    def transform(self, data: U_contra, /) -> V_co: ...


@runtime_checkable
class InvertibleTransform(Transform[U, V], Protocol):
    """Protocol for invertible transformers."""

    @abstractmethod
    def inverse_transform(self, data: V, /) -> U: ...


@runtime_checkable
class EncoderProtocol(Protocol[U, V]):
    """Minimal Protocol for Encoders.

    This protocol should be used in applications that only use encoders, but do not need to
    worry about creating new encoders or chaining them together.
    """

    @abstractmethod
    def fit(self, data: U, /) -> None: ...
    @abstractmethod
    def encode(self, data: U, /) -> V: ...
    @abstractmethod
    def decode(self, data: V, /) -> U: ...


class Encoder(EncoderProtocol[U, V], Protocol):
    """Protocol for Encoders with algebraic mixin methods."""

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

    def __pow__(self: "Encoder[T, T]", power: int, /) -> "Encoder[T, T]":
        r"""Return the chain of itself multiple times.

        Example:
            enc = self ** n
            enc(x) == self(self(...self(x)...))
        """
        return pow_encoder(self, power)

    # endregion mixin methods ----------------------------------------------------------


E = TypeVar("E", bound=Encoder)
"""Type alias for Encoder."""


class BaseEncoder(Encoder[T, T2]):
    r"""Base class that all encoders must subclass."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the Encoder."""

    _is_fitted: bool = False
    r"""Whether the encoder has been fitted."""

    # region abstract methods ----------------------------------------------------------
    @property
    @abstractmethod
    def requires_fit(self) -> bool:
        r"""Whether the encoder requires fitting."""
        ...

    @abstractmethod
    def encode(self, data: T, /) -> T2:
        r"""Encode the data by transformation."""
        ...

    @abstractmethod
    def decode(self, data: T2, /) -> T:
        r"""Decode the data by inverse transformation."""
        ...

    def fit(self, data: T, /) -> None:
        r"""Implement as necessary."""

    def simplify(self) -> Encoder[T, T2]:
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
        def fit(self: Self, data: T, /) -> None:
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
        def encode(self: Self, data: T, /) -> T2:
            r"""Encode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted.")
            return original_encode(self, data)

        @wraps(original_decode)
        def decode(self: Self, data: T2, /) -> T:
            r"""Decode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted.")
            return original_decode(self, data)

        cls.fit = fit  # type: ignore[method-assign]
        cls.encode = encode  # type: ignore[method-assign]
        cls.decode = decode  # type: ignore[method-assign]

    @property
    def is_fitted(self) -> bool:
        r"""Whether the encoder has been fitted."""
        return (not self.requires_fit) or self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool, /) -> None:
        self._is_fitted = value

    @property
    def is_surjective(self) -> bool:
        r"""Whether the encoder is surjective."""
        raise NotImplementedError

    @property
    def is_injective(self) -> bool:
        r"""Whether the encoder is injective."""
        raise NotImplementedError

    @property
    def is_bijective(self) -> bool:
        r"""Whether the encoder is bijective."""
        return self.is_surjective and self.is_injective

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        r"""Return the parameters of the encoder."""
        return self.__dict__

    def set_params(self, **kwargs: Any) -> None:
        r"""Set the parameters of the encoder."""
        self.__dict__.update(kwargs)

    # region method aliases ------------------------------------------------------------
    def fit_transform(self, data: T, /) -> T2:
        r"""Fit the encoder to the data and apply the transformation."""
        self.fit(data)
        return self.encode(data)

    def transform(self, data: T, /) -> T2:
        r"""Alias for encode."""
        return self.encode(data)

    def inverse_transform(self, data: T2, /) -> T:
        r"""Alias for decode."""
        return self.decode(data)

    # endregion method aliases ---------------------------------------------------------

    # region chaining methods ----------------------------------------------------------
    # def standardize(self) -> "ChainedEncoder[Self, StandardScaler]":
    #     r"""Chain a standardize."""
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


class IdentityEncoder(BaseEncoder):
    r"""Dummy class that performs identity function."""

    requires_fit: ClassVar[bool] = False
    is_injective: ClassVar[bool] = True
    is_surjective: ClassVar[bool] = True
    is_bijective: ClassVar[bool] = True

    def encode(self, data: T, /) -> T:
        return data

    def decode(self, data: T, /) -> T:
        return data


class TupleEncoder(BaseEncoder):
    r"""Wraps input into a tuple."""

    requires_fit: ClassVar[bool] = False
    is_injective: ClassVar[bool] = True
    is_surjective: ClassVar[bool] = True
    is_bijective: ClassVar[bool] = True

    def __invert__(self) -> "TupleDecoder":
        return TupleDecoder()

    def encode(self, data: T, /) -> tuple[T]:
        return (data,)

    def decode(self, data: tuple[T], /) -> T:
        return data[0]


class TupleDecoder(BaseEncoder):
    r"""Unwraps input from a tuple."""

    requires_fit: ClassVar[bool] = False
    is_injective: ClassVar[bool] = True
    is_surjective: ClassVar[bool] = True
    is_bijective: ClassVar[bool] = True

    def __invert__(self) -> "TupleEncoder":
        return TupleEncoder()

    def encode(self, data: tuple[T], /) -> T:
        return data[0]

    def decode(self, data: T, /) -> tuple[T]:
        return (data,)


class CopyEncoder(BaseEncoder[T, T]):
    r"""Encoder that deepcopies the input."""

    requires_fit: ClassVar[bool] = False
    is_injective: ClassVar[bool] = True
    is_surjective: ClassVar[bool] = True
    is_bijective: ClassVar[bool] = True

    def encode(self, data: T, /) -> T:
        return deepcopy(data)

    def decode(self, data: T, /) -> T:
        return deepcopy(data)


class InverseEncoder(BaseEncoder[T, T2]):
    """Applies an encoder in reverse."""

    encoder: Encoder[T2, T]
    """The encoder to invert."""

    def __init__(self, encoder: Encoder[T2, T], /) -> None:
        self.encoder = encoder

    def __repr__(self) -> str:
        return f"~{self.encoder}"

    def fit(self, data: T, /) -> None:
        raise NotImplementedError("Inverse encoders cannot be fitted.")

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def encode(self, data: T, /) -> T2:
        return self.encoder.decode(data)

    def decode(self, data: T2, /) -> T:
        return self.encoder.encode(data)

    def simplify(self) -> Self:
        cls = type(self)
        return cls(self.encoder.simplify())


def invert_encoder(encoder: Encoder[T, T2], /) -> Encoder[T2, T]:
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
                raise ValueError(f"Index {index} not supported.")

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

    @property
    def is_surjective(self) -> bool:
        return all(e.is_surjective for e in self.encoders)  # type: ignore[attr-defined]

    @property
    def is_injective(self) -> bool:
        return all(e.is_injective for e in self.encoders)  # type: ignore[attr-defined]

    def fit(self, data: Any, /) -> None:
        for encoder in reversed(self.encoders):
            try:
                encoder.fit(data)
            except Exception as exc:
                raise RuntimeError(f"Failed to fit {type(encoder)}") from exc
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
                raise ValueError(f"Index {index} not supported.")

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

    @property
    def is_surjective(self) -> bool:
        return all(e.is_surjective for e in self.encoders)  # type: ignore[attr-defined]

    @property
    def is_injective(self) -> bool:
        return all(e.is_injective for e in self.encoders)  # type: ignore[attr-defined]

    def fit(self, data: Any, /) -> None:
        for encoder in reversed(self.encoders):
            try:
                encoder.fit(data)
            except Exception as exc:
                raise RuntimeError(f"Failed to fit {type(encoder)}") from exc
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

    @property
    def is_surjective(self) -> bool:
        return all(e.is_surjective for e in self.encoders)  # type: ignore[attr-defined]

    @property
    def is_injective(self) -> bool:
        return all(e.is_injective for e in self.encoders)  # type: ignore[attr-defined]

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
                raise ValueError(f"Index {index} not supported.")

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


class DuplicateEncoder(BaseEncoder[tuple[T, ...], tuple[T2, ...]]):
    r"""Duplicate encoder multiple times (references same object)."""

    base_encoder: Encoder[T, T2]

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def __init__(self, encoder: Encoder[T, T2], n: int = 1) -> None:
        self.base_encoder = encoder
        self.n = n
        self.encoder = ParallelEncoder(*(self.base_encoder for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: tuple[T, ...], /) -> None:
        return self.encoder.fit(data)

    def encode(self, data: tuple[T, ...], /) -> tuple[T2, ...]:
        return self.encoder.encode(data)

    def decode(self, data: tuple[T2, ...], /) -> tuple[T, ...]:
        return self.encoder.decode(data)


class CloneEncoder(BaseEncoder[tuple[T, ...], tuple[T2, ...]]):
    r"""Clone encoder multiple times (distinct copies)."""

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def __init__(self, encoder: BaseEncoder, n: int = 1) -> None:
        self.base_encoder = encoder
        self.n = n
        self.encoder = ParallelEncoder(*(deepcopy(self.base_encoder) for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: tuple[T, ...], /) -> None:
        return self.encoder.fit(data)

    def encode(self, data: tuple[T, ...], /) -> tuple[T2, ...]:
        return self.encoder.encode(data)

    def decode(self, data: tuple[T2, ...], /) -> tuple[T, ...]:
        return self.encoder.decode(data)
