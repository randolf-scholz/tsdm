r"""Base Classes for Encoders."""

__all__ = [
    # Types / TypeVars
    "Transform",
    "InvertibleTransform",
    "EncoderProtocol",
    # ABCs
    "Encoder",
    # Classes
    "BaseEncoder",
    "ChainedEncoder",
    "CloneEncoder",
    "CopyEncoder",
    "DuplicateEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "MappingEncoder",
    "PipedEncoder",
    "ParallelEncoder",
    # Functions
    "chain_encoders",
    "parallel_encoders",
    "duplicate_encoder",
    "invert_encoder",
    "pipe_encoders",
    "pow_encoder",
]
# TODO: Improve Typing for Encoders.

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

E = TypeVar("E", bound="Encoder")
"""Type alias for encoder_var."""

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
    """Minimal Protocol for Encoders."""

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
    def simplify(self) -> Self:
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
        return InverseEncoder(self)  # type: ignore[arg-type]

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
            >>> enc = other @ self
            >>> enc(x) == other(self(x))
        """
        return ChainedEncoder(other, self)

    def __rshift__(self, other: "Encoder[V, W]", /) -> "Encoder[U, W]":
        r"""Pipe the encoders (encoder composition).

        Note that the order is reversed compared to `@`.

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


class BaseEncoderMetaClass(type(Protocol)):  # type: ignore[misc]
    r"""Metaclass for BaseDataset."""

    def __init__(
        self, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            self.LOGGER = logging.getLogger(f"{self.__module__}.{self.__name__}")


class BaseEncoder(Encoder[T, T2], metaclass=BaseEncoderMetaClass):
    r"""Base class that all encoders must subclass."""

    LOGGER: ClassVar[logging.Logger]
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

    def simplify(self) -> Self:
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
        def fit(self: BaseEncoder, data: T, /) -> None:
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
        def encode(self: BaseEncoder, data: T, /) -> T2:
            r"""Encode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted.")
            return original_encode(self, data)

        @wraps(original_decode)
        def decode(self: BaseEncoder, data: T2, /) -> T:
            r"""Decode the data."""
            if self.requires_fit and not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted.")
            return original_decode(self, data)

        cls.fit = fit  # type: ignore[method-assign]
        cls.encode = encode  # type: ignore[method-assign]
        cls.decode = decode  # type: ignore[method-assign]
        if not hasattr(cls, "transform"):
            cls.transform = cls.encode
        if not hasattr(cls, "inverse_transform"):
            cls.inverse_transform = cls.decode

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


class InverseEncoder(BaseEncoder[T2, T]):
    """Applies an encoder in reverse."""

    encoder: BaseEncoder[T, T2]
    """The encoder to invert."""

    def __init__(self, encoder: BaseEncoder[T, T2], /) -> None:
        self.encoder = encoder

    def __repr__(self) -> str:
        return f"~{self.encoder}"

    def fit(self, data: T2, /) -> None:
        raise NotImplementedError("Inverse encoders cannot be fitted.")

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def encode(self, data: T2, /) -> T:
        return self.encoder.decode(data)

    def decode(self, data: T, /) -> T2:
        return self.encoder.encode(data)

    def simplify(self) -> Self:
        cls = type(self)
        return cls(self.encoder.simplify())


def invert_encoder(encoder: Encoder[T, T2], /) -> Encoder[T2, T]:
    r"""Return the inverse encoder (i.e. decoder)."""
    return ~encoder


@pprint_repr(recursive=2)
class ChainedEncoder(BaseEncoder, Sequence[E]):
    r"""Represents function composition of encoders."""

    encoders: list[E]
    r"""List of encoders."""

    def __init__(self, *encoders: E, simplify: bool = True) -> None:
        self.encoders = []
        for encoder in encoders:
            if simplify and isinstance(encoder, ChainedEncoder):
                self.encoders.extend(encoder)
            else:
                self.encoders.append(encoder)

    def __invert__(self) -> Self:
        cls = type(self)
        return cls(*(~e for e in reversed(self.encoders)))  # type: ignore[arg-type]

    def __len__(self) -> int:
        r"""Return number of chained encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> E: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    def __getitem__(self, index):
        r"""Get the encoder at the given index."""
        if isinstance(index, int):
            return self.encoders[index]
        if isinstance(index, slice):
            return ChainedEncoder(*self.encoders[index])
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

    def simplify(self) -> IdentityEncoder | E | Self:  # type: ignore[override]
        r"""Simplify the chained encoder."""
        if len(self) == 0:
            return IdentityEncoder()
        if len(self) == 1:
            encoder = self[0]
            return encoder.simplify()
        cls = type(self)
        return cls(*(e.simplify() for e in self))


@overload
def chain_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
@overload
def chain_encoders(e: E, /, *, simplify: Literal[True]) -> E: ...
@overload
def chain_encoders(*encoders: E, simplify: bool = ...) -> ChainedEncoder[E]: ...
def chain_encoders(*encoders, simplify=True):
    r"""Chain encoders."""
    if len(encoders) == 0 and simplify:
        return IdentityEncoder()
    if len(encoders) == 1 and simplify:
        return encoders[0]
    return ChainedEncoder(*encoders, simplify=simplify)


@pprint_repr(recursive=2)
class PipedEncoder(BaseEncoder, Sequence[E]):
    r"""Represents function composition of encoders."""

    encoders: list[E]
    r"""List of encoders."""

    def __init__(self, *encoders: E, simplify: bool = True) -> None:
        self.encoders = []
        for encoder in encoders:
            if simplify and isinstance(encoder, PipedEncoder):
                self.encoders.extend(encoder)
            else:
                self.encoders.append(encoder)

    def __invert__(self) -> Self:
        cls = type(self)
        return cls(*(~e for e in reversed(self.encoders)))  # type: ignore[arg-type]

    def __len__(self) -> int:
        r"""Return number of chained encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> E: ...
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

    def simplify(self) -> IdentityEncoder | E | Self:  # type: ignore[override]
        r"""Simplify the chained encoder."""
        if len(self) == 0:
            return IdentityEncoder()
        if len(self) == 1:
            encoder = self[0]
            return encoder.simplify()
        cls = type(self)
        return cls(*(e.simplify() for e in self), simplify=True)


@overload
def pipe_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
@overload
def pipe_encoders(e: E, /, *, simplify: Literal[True]) -> E: ...
@overload
def pipe_encoders(*encoders: E, simplify: bool = ...) -> PipedEncoder[E]: ...
def pipe_encoders(*encoders, simplify=True):
    r"""Chain encoders."""
    if len(encoders) == 0 and simplify:
        return IdentityEncoder()
    if len(encoders) == 1 and simplify:
        return encoders[0]
    return PipedEncoder(*encoders, simplify=simplify)


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
    e: E, n: Literal[1], /, *, simplify: Literal[True], copy: bool = ...
) -> E: ...
@overload
def pow_encoder(
    e: E, n: int, /, *, simplify: bool = ..., copy: bool = ...
) -> PipedEncoder[E]: ...
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
class ParallelEncoder(BaseEncoder[tuple[Any, ...], tuple[Any, ...]], Sequence[E]):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """

    encoders: list[E]
    r"""The encoders."""

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

    def __init__(self, *encoders: E, simplify: bool = True) -> None:
        self.encoders = []

        for encoder in encoders:
            if simplify and isinstance(encoder, ParallelEncoder):
                self.encoders.extend(encoder)
            else:
                self.encoders.append(encoder)

    def __len__(self) -> int:
        r"""Return the number of the encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> E: ...
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

    def simplify(self) -> IdentityEncoder | E | Self:  # type: ignore[override]
        r"""Simplify the product encoder."""
        if len(self.encoders) == 0:
            return IdentityEncoder()
        if len(self.encoders) == 1:
            enocder = self.encoders[0]
            return enocder.simplify()
        cls = type(self)
        return cls(*(e.simplify() for e in self.encoders))


@overload
def parallel_encoders(*, simplify: Literal[True]) -> IdentityEncoder: ...
@overload
def parallel_encoders(e: E, /, *, simplify: Literal[True]) -> E: ...
@overload
def parallel_encoders(
    e1: E, e2: E, /, *encoders: E, simplify: bool = ...
) -> ParallelEncoder[E]: ...
def parallel_encoders(*encoders, simplify=True):
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """
    if len(encoders) == 0 and simplify:
        return IdentityEncoder()
    if len(encoders) == 1 and simplify:
        return encoders[0]
    return ParallelEncoder(*encoders, simplify=simplify)


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
    e: E, n: Literal[1], /, *, simplify: Literal[True], copy: bool = ...
) -> E: ...
@overload
def duplicate_encoder(
    e: E, n: int, /, *, simplify: bool = ..., copy: bool = ...
) -> ParallelEncoder[E]: ...
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
class MappingEncoder(BaseEncoder[Mapping[K, Any], Mapping[K, Any]], Mapping[K, E]):
    r"""Encoder that maps keys to encoders."""

    encoders: Mapping[K, E]
    r"""Mapping of keys to encoders."""

    @property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self.encoders.values())

    def __init__(self, encoders: Mapping[K, E], /) -> None:
        self.encoders = encoders

    @overload
    def __getitem__(self, key: list[K], /) -> Self: ...
    @overload
    def __getitem__(self, key: K, /) -> E: ...
    def __getitem__(self, key, /):
        r"""Get the encoder for the given key."""
        if isinstance(key, list):
            return MappingEncoder({k: self.encoders[k] for k in key})
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

    def simplify(self) -> IdentityEncoder | E | Self:  # type: ignore[override]
        r"""Simplify the mapping encoder."""
        if len(self.encoders) == 0:
            return IdentityEncoder()
        if len(self.encoders) == 1:
            encoder = next(iter(self.encoders.values()))
            return encoder.simplify()
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
