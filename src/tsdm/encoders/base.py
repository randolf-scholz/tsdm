r"""Base Classes for Encoders."""

from __future__ import annotations

__all__ = [
    # Types / TypeVars
    "Encoder",
    "encoder_var",
    # Classes
    "BaseEncoder",
    "ChainedEncoder",
    "CloneEncoder",
    "DuplicateEncoder",
    "IdentityEncoder",
    "InverseEncoder",
    "MappingEncoder",
    "ProductEncoder",
    # Functions
    "chain_encoders",
    "direct_sum_encoders",
    "duplicate_encoder",
    "invert_encoder",
    "pow_encoder",
]
# TODO: Improve Typing for Encoders.

import logging
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from functools import wraps

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

from tsdm.types.variables import any_other_var as S, any_var as T, key_var as K
from tsdm.utils.strings import repr_object, repr_type

encoder_var = TypeVar("encoder_var", bound="BaseEncoder")
"""Type variable for encoders."""

E = TypeVar("E", bound="BaseEncoder")
"""Type alias for encoder_var."""
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")
Y = TypeVar("Y")


@runtime_checkable
class Encoder(Protocol[U, V]):
    """Protocol for Encoders."""

    @property
    def is_fitted(self) -> bool:
        """Whether the encoder has been fitted."""
        ...

    @property
    def requires_fit(self) -> bool:
        r"""Whether the encoder requires fitting."""
        ...

    def __invert__(self) -> Encoder[V, U]:
        r"""Return the inverse encoder (i.e. decoder)."""
        ...

    def __matmul__(self, other: Encoder[X, U], /) -> Encoder[X, V]:
        r"""Chain the encoders (pure function composition).

        Example:
            enc = enc1 @ enc2
            enc(x) == enc1(enc2(x))

        Raises:
            TypeError if other is not an encoder.
        """
        ...

    def __gt__(self, other: Encoder[V, W], /) -> Encoder[U, W]:
        r"""Pipe the encoders (encoder composition).

        Note that the order is reversed compared to `@`.

        Example:
            enc = enc1 > enc2
            enc.encode(x) == enc2.encode(enc1.encode(x))
            enc.decode(y) == enc1.decode(enc2.decode(y))

        Note:
            - `>` is associative:
                (enc1 > enc2) > enc3 == enc1 > (enc2 > enc3)
              Proof::
                 ((A > B) > C).encode(x)
                     = C.encode((A > B).encode(x))
                     = C.encode(B.encode(A.encode(x)))
                 (A > (B > C)).encode(x)
                     = (B > C).encode(A.encode(x))
                     = C.encode(B.encode(A.encode(x)))
            - inverse law:
              ~(enc1 > enc2) == ~enc2 > ~enc1
              Proof::
                  ~(A > B).encode(x)
                      = (A > B).decode(x)
                      = B.decode(A.decode(x))
        """
        return other @ self

    # FIXME: Encoder[tuple[U, X], tuple[V, Y]] causes segmentation fault
    def __or__(self, other: Encoder[X, Y], /) -> Encoder:
        r"""Return product encoders."""
        ...

    def encode(self, data: U, /) -> V:
        """Encode the data by transformation."""
        ...

    def decode(self, data: V, /) -> U:
        """Decode the data by transformation."""
        ...

    def fit(self, data: U, /) -> None:
        r"""Fits the encoder to data."""
        ...


class BaseEncoderMetaClass(type(Protocol)):  # type: ignore[misc]
    r"""Metaclass for BaseDataset."""

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")


class BaseEncoder(Encoder[T, S], metaclass=BaseEncoderMetaClass):
    r"""Base class that all encoders must subclass."""

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the Encoder."""

    _is_fitted: bool = False
    r"""Whether the encoder has been fitted."""

    def __init_subclass__(cls) -> None:
        r"""Initialize the subclass.

        The wrapping of fit/encode/decode must be done here to avoid `~pickle.PickleError`!
        """
        original_fit = cls.fit
        original_encode = cls.encode
        original_decode = cls.decode

        @wraps(original_fit)
        def fit(self: BaseEncoder, data: T, /) -> None:
            r"""Fit the encoder to the data."""
            if not self.requires_fit:
                self.LOGGER.info(
                    "Skipping fitting as encoder does not require fitting."
                )
            else:
                self.LOGGER.info("Fitting encoder to data.")
                original_fit(self, data)
            self.is_fitted = True

        @wraps(original_encode)
        def encode(self: BaseEncoder, data: T, /) -> S:
            r"""Encode the data."""
            if not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted.")
            return original_encode(self, data)

        @wraps(original_decode)
        def decode(self: BaseEncoder, data: S, /) -> T:
            r"""Decode the data."""
            if not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted.")
            return original_decode(self, data)

        cls.fit = fit  # type: ignore[method-assign]
        cls.encode = encode  # type: ignore[method-assign]
        cls.decode = decode  # type: ignore[method-assign]

    def __init__(self) -> None:
        super().__init__()
        self.transform = self.encode
        self.inverse_transform = self.decode

    def __invert__(self) -> BaseEncoder[S, T]:
        r"""Return the inverse encoder (i.e. decoder)."""
        return InverseEncoder(self)

    def __matmul__(self, other: Encoder, /) -> ChainedEncoder:
        r"""Return chained encoders."""
        return ChainedEncoder(self, other)

    def __or__(self, other: Encoder, /) -> ProductEncoder:
        r"""Return product encoders."""
        return ProductEncoder(self, other)

    def __pow__(self, power: int) -> DuplicateEncoder:
        r"""Return the product encoder of the encoder with itself power many times."""
        return DuplicateEncoder(self, power)

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        return repr_object(self, fallback=repr_type, recursive=1)

    @property
    @abstractmethod
    def requires_fit(self) -> bool:
        r"""Whether the encoder requires fitting."""
        ...

    @property
    def is_fitted(self) -> bool:
        r"""Whether the encoder has been fitted."""
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
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

    def fit(self, data: T, /) -> None:
        r"""Implement as necessary."""

    @abstractmethod
    def encode(self, data: T, /) -> S:
        r"""Encode the data by transformation."""
        ...

    @abstractmethod
    def decode(self, data: S, /) -> T:
        r"""Decode the data by inverse transformation."""
        ...

    def simplify(self) -> Self:
        r"""Simplify the encoder."""
        return self

    def _post_fit_hook(self) -> None:
        self.is_fitted = True

    def _pre_encode_hook(self) -> None:
        if self.requires_fit and not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")

    def _pre_decode_hook(self) -> None:
        if self.requires_fit and not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")


class IdentityEncoder(BaseEncoder[T, T]):
    r"""Dummy class that performs identity function."""

    requires_fit: ClassVar[bool] = False
    is_injective: ClassVar[bool] = True
    is_surjective: ClassVar[bool] = True
    is_bijective: ClassVar[bool] = True

    def encode(self, data: T, /) -> T:
        return data

    def decode(self, data: T, /) -> T:
        return data


class InverseEncoder(BaseEncoder[S, T]):
    """Applies an encoder in reverse."""

    encoder: BaseEncoder[T, S]
    """The encoder to invert."""

    def __init__(self, encoder: BaseEncoder[T, S], /) -> None:
        super().__init__()
        self.encoder = encoder

    def __repr__(self) -> str:
        return f"~{self.encoder}"

    def fit(self, data: S, /) -> None:
        raise NotImplementedError("Inverse encoders cannot be fitted.")

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def encode(self, data: S, /) -> T:
        return self.encoder.decode(data)

    def decode(self, data: T, /) -> S:
        return self.encoder.encode(data)

    def simplify(self) -> Self:
        cls = type(self)
        return cls(self.encoder.simplify())


def invert_encoder(encoder: BaseEncoder[T, S], /) -> BaseEncoder[S, T]:
    r"""Return the inverse encoder (i.e. decoder)."""
    return ~encoder


class ChainedEncoder(BaseEncoder, Sequence[E]):
    r"""Represents function composition of encoders."""

    encoders: list[E]
    r"""List of encoders."""

    def __init__(self, *encoders: E, simplify: bool = True) -> None:
        super().__init__()
        self.encoders = []
        for encoder in encoders:
            if simplify and isinstance(encoder, ChainedEncoder):
                self.encoders.extend(encoder)
            else:
                self.encoders.append(encoder)

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        # One more level of recursion than BaseEncoder.__repr__
        return repr_object(self, fallback=repr_type, recursive=2)

    def __invert__(self) -> Self:
        cls = type(self)
        return cls(*(~e for e in reversed(self.encoders)))  # type: ignore[arg-type]

    def __len__(self) -> int:
        r"""Return number of chained encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> E: ...
    @overload
    def __getitem__(self, index: slice) -> ChainedEncoder[E]: ...
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
        return all(e.is_surjective for e in self.encoders)

    @property
    def is_injective(self) -> bool:
        return all(e.is_injective for e in self.encoders)

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
def chain_encoders(*, simplify: Literal[True] = True) -> IdentityEncoder: ...
@overload
def chain_encoders(e: E, /, *, simplify: Literal[True] = True) -> E: ...
@overload
def chain_encoders(
    e1: E,
    e2: E,
    /,
    *encoders: E,
    simplify: Literal[True] = True,
) -> ChainedEncoder[E]: ...
@overload
def chain_encoders(*encoders: E, simplify: Literal[False]) -> ChainedEncoder[E]: ...
def chain_encoders(*encoders, simplify=True):
    r"""Chain encoders."""
    if len(encoders) == 0 and simplify:
        return IdentityEncoder()
    if len(encoders) == 1 and simplify:
        return encoders[0]
    return ChainedEncoder(*encoders, simplify=simplify)


@overload
def pow_encoder(
    e: E,
    n: Literal[0],
    /,
    *,
    simplify: Literal[True] = True,
    copy: bool = True,
) -> IdentityEncoder: ...
@overload
def pow_encoder(
    e: E,
    n: Literal[1],
    /,
    *,
    simplify: Literal[True] = True,
    copy: bool = True,
) -> E: ...
@overload
def pow_encoder(
    e: E, n: int, /, *, simplify: Literal[False], copy: bool = True
) -> ChainedEncoder[E]: ...
def pow_encoder(
    encoder: E, n: int, /, *, simplify: bool = True, copy: bool = True
) -> Encoder:
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
        return chain_encoders(*[~e for e in reversed(encoders)])
    return chain_encoders(*encoders)


class ProductEncoder(BaseEncoder, Sequence[E]):
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
        return all(e.is_surjective for e in self.encoders)

    @property
    def is_injective(self) -> bool:
        return all(e.is_injective for e in self.encoders)

    def __init__(self, *encoders: E, simplify: bool = True) -> None:
        super().__init__()
        self.encoders = []

        for encoder in encoders:
            if simplify and isinstance(encoder, ProductEncoder):
                self.encoders.extend(encoder)
            else:
                self.encoders.append(encoder)

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        # One more level of recursion than BaseEncoder.__repr__
        return repr_object(self, fallback=repr_type, recursive=2)

    def __len__(self) -> int:
        r"""Return the number of the encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> E: ...
    @overload
    def __getitem__(self, index: slice) -> ProductEncoder[E]: ...
    def __getitem__(self, index: int | slice) -> E | ProductEncoder[E]:
        r"""Get the encoder at the given index."""
        if isinstance(index, int):
            return self.encoders[index]
        if isinstance(index, slice):
            return ProductEncoder(*self.encoders[index])
        raise ValueError(f"Index {index} not supported.")

    def fit(self, data: tuple[Any, ...], /) -> None:
        for encoder, x in zip(self.encoders, data):
            encoder.fit(x)

    def encode(self, data: tuple[Any, ...], /) -> tuple[Any, ...]:
        rtype = type(data)
        return rtype(encoder.encode(x) for encoder, x in zip(self.encoders, data))

    def decode(self, data: tuple[Any, ...], /) -> tuple[Any, ...]:
        rtype = type(data)
        return rtype(encoder.decode(x) for encoder, x in zip(self.encoders, data))

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
def direct_sum_encoders(*, simplify: Literal[True] = True) -> IdentityEncoder: ...
@overload
def direct_sum_encoders(e: E, /, *, simplify: Literal[True] = True) -> E: ...
@overload
def direct_sum_encoders(
    e1: E,
    e2: E,
    /,
    *encoders: E,
    simplify: Literal[False],
) -> ProductEncoder[E]: ...
def direct_sum_encoders(
    *encoders: E, simplify: bool = True, copy: bool = True
) -> Encoder:
    r"""Product-Type for Encoders.

    Applies multiple encoders in parallel on tuples of data.
    """
    if len(encoders) == 0 and simplify:
        return IdentityEncoder()
    if len(encoders) == 1 and simplify:
        return encoders[0]
    return ProductEncoder(*encoders, simplify=simplify)


@overload
def duplicate_encoder(
    e: E,
    n: Literal[0],
    /,
    *,
    simplify: Literal[True] = True,
    copy: bool = True,
) -> IdentityEncoder: ...
@overload
def duplicate_encoder(
    e: E,
    n: Literal[1],
    /,
    *,
    simplify: Literal[True] = True,
    copy: bool = True,
) -> E: ...
@overload
def duplicate_encoder(
    e: E, n: int, /, *, simplify: Literal[False], copy: bool = True
) -> ProductEncoder[E]: ...
def duplicate_encoder(
    encoder: E, n: int, /, *, simplify: bool = True, copy: bool = True
) -> Encoder:
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
        return direct_sum_encoders(*[~e for e in reversed(encoders)])
    return direct_sum_encoders(*encoders)


class MappingEncoder(BaseEncoder, Mapping[K, E]):
    r"""Encoder that maps keys to encoders."""

    encoders: Mapping[K, E]
    r"""Mapping of keys to encoders."""

    @property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self.encoders.values())

    def __init__(self, encoders: Mapping[K, E]) -> None:
        super().__init__()
        self.encoders = encoders

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        # One more level of recursion than BaseEncoder.__repr__
        return repr_object(self, fallback=repr_type, recursive=2)

    @overload
    def __getitem__(self, key: K) -> E: ...
    @overload
    def __getitem__(self, key: list[K]) -> MappingEncoder[K, E]: ...
    def __getitem__(self, key: K | list[K]) -> E | MappingEncoder[K, E]:
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


class DuplicateEncoder(BaseEncoder[tuple[T, ...], tuple[S, ...]]):
    r"""Duplicate encoder multiple times (references same object)."""

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def __init__(self, encoder: BaseEncoder, n: int = 1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(self.base_encoder for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: tuple[T, ...], /) -> None:
        return self.encoder.fit(data)

    def encode(self, data: tuple[T, ...], /) -> tuple[S, ...]:
        return self.encoder.encode(data)

    def decode(self, data: tuple[S, ...], /) -> tuple[T, ...]:
        return self.encoder.decode(data)


class CloneEncoder(BaseEncoder[tuple[T, ...], tuple[S, ...]]):
    r"""Clone encoder multiple times (distinct copies)."""

    @property
    def requires_fit(self) -> bool:
        return self.encoder.requires_fit

    def __init__(self, encoder: BaseEncoder, n: int = 1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(deepcopy(self.base_encoder) for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: tuple[T, ...], /) -> None:
        return self.encoder.fit(data)

    def encode(self, data: tuple[T, ...], /) -> tuple[S, ...]:
        return self.encoder.encode(data)

    def decode(self, data: tuple[S, ...], /) -> tuple[T, ...]:
        return self.encoder.decode(data)
