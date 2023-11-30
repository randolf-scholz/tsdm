#!/usr/bin/env python

"""This is a test file for mypy."""

# from __future__ import annotations

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

encoder_var = TypeVar("encoder_var", bound="BaseEncoder")
"""Type variable for encoders."""

E = TypeVar("E", bound="BaseEncoder")
"""Type alias for encoder_var."""
T = TypeVar("T")
S = TypeVar("S")
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

    def __invert__(self) -> "Encoder[V, U]":
        r"""Return the inverse encoder (i.e. decoder)."""
        ...

    def __matmul__(self, other: "Encoder[X, U]", /) -> "Encoder[X, V]":
        r"""Chain the encoders (pure function composition).

        Example:
            enc = enc1 @ enc2
            enc(x) == enc1(enc2(x))

        Raises:
            TypeError if other is not an encoder.
        """
        ...

    def __gt__(self, other: "Encoder[V, W]", /) -> "Encoder[U, W]":
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

    def __or__(self, other: "Encoder[X, Y]", /) -> "Encoder[tuple[U, X], tuple[V, Y]]":
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
        super().__init_subclass__()
        original_fit = cls.fit
        original_encode = cls.encode
        original_decode = cls.decode

        @wraps(original_fit)
        def fit(self: Self, data: T, /) -> None:
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
        def encode(self: Self, data: T, /) -> S:
            r"""Encode the data."""
            if not self.is_fitted:
                raise RuntimeError("Encoder has not been fitted.")
            return original_encode(self, data)

        @wraps(original_decode)
        def decode(self: Self, data: S, /) -> T:
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


x = IdentityEncoder.__parameters__
print(x)
