r"""Base Classes for Encoders."""


from __future__ import annotations

__all__ = [
    # TypeVars
    "EncoderVar",
    # Classes
    "BaseEncoder",
    "IdentityEncoder",
    "ChainedEncoder",
    "ProductEncoder",
    "DuplicateEncoder",
    "CloneEncoder",
    "MappingEncoder",
]
# TODO: Improve Typing for Encoders.

import logging
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar, Generic, Iterator, TypeVar, overload

from tsdm.utils.decorators import wrap_method
from tsdm.utils.strings import repr_object
from tsdm.utils.types import KeyVar, ObjectVar, ReturnVar


class BaseEncoderMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


class BaseEncoder(ABC, Generic[ObjectVar, ReturnVar], metaclass=BaseEncoderMetaClass):
    r"""Base class that all encoders must subclass."""

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the Encoder."""

    requires_fit: ClassVar[bool] = True
    r"""Whether the encoder requires fitting."""

    _is_fitted: bool = False
    r"""Whether the encoder has been fitted."""

    def __init__(self) -> None:
        super().__init__()
        self.transform = self.encode
        self.inverse_transform = self.decode

    def __init_subclass__(cls, /, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the subclass.

        The wrapping of fit/encode/decode must be done here to avoid
        `~pickle.PickleError`!
        """
        super().__init_subclass__(*args, **kwargs)
        cls.fit = wrap_method(cls.fit, after=cls._post_fit_hook)  # type: ignore[assignment]
        cls.encode = wrap_method(cls.encode, before=cls._pre_encode_hook)  # type: ignore[assignment]
        cls.decode = wrap_method(cls.decode, before=cls._pre_decode_hook)  # type: ignore[assignment]

    # TODO: implement __invert__ (python 3.11)

    def __matmul__(self, other: BaseEncoder) -> ChainedEncoder:
        r"""Return chained encoders."""
        return ChainedEncoder(self, other)

    def __or__(self, other: BaseEncoder) -> ProductEncoder:
        r"""Return product encoders."""
        return ProductEncoder(self, other)

    def __pow__(self, power: int) -> DuplicateEncoder:
        r"""Return the product encoder of the encoder with itself power many times."""
        return DuplicateEncoder(self, power)

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        return repr_object(self)

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
        return False

    @property
    def is_injective(self) -> bool:
        r"""Whether the encoder is injective."""
        return True

    @property
    def is_bijective(self) -> bool:
        r"""Whether the encoder is bijective."""
        return self.is_surjective and self.is_injective

    def fit(self, data: ObjectVar, /) -> None:
        r"""Implement as necessary."""

    @abstractmethod
    def encode(self, data: ObjectVar, /) -> ReturnVar:
        r"""Encode the data by transformation."""

    @abstractmethod
    def decode(self, data: ObjectVar, /) -> ReturnVar:
        r"""Decode the data by inverse transformation."""

    def _post_fit_hook(self) -> None:
        self.is_fitted = True

    def _pre_encode_hook(self) -> None:
        if self.requires_fit and not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")

    def _pre_decode_hook(self) -> None:
        if self.requires_fit and not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")


EncoderVar = TypeVar("EncoderVar", bound=BaseEncoder)
"""Type variable for encoders."""


class IdentityEncoder(BaseEncoder):
    r"""Dummy class that performs identity function."""

    requires_fit: ClassVar[bool] = False

    def encode(self, data: ObjectVar, /) -> ObjectVar:
        return data

    def decode(self, data: ObjectVar, /) -> ObjectVar:
        return data


class ProductEncoder(BaseEncoder, Sequence[EncoderVar]):
    r"""Product-Type for Encoders."""

    requires_fit: bool = True  # type: ignore[misc]

    encoders: list[EncoderVar]
    r"""The encoders."""

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

    def __init__(self, *encoders: EncoderVar, simplify: bool = True) -> None:
        super().__init__()
        self.encoders = []

        for encoder in encoders:
            if simplify and isinstance(encoder, ProductEncoder):
                for enc in encoder:
                    self.encoders.append(enc)
            else:
                self.encoders.append(encoder)
        self.requires_fit = any(e.requires_fit for e in self.encoders)

    def __len__(self) -> int:
        r"""Return the number of the encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> EncoderVar:
        ...

    @overload
    def __getitem__(self, index: slice) -> ProductEncoder[EncoderVar]:
        ...

    def __getitem__(
        self, index: int | slice
    ) -> EncoderVar | ProductEncoder[EncoderVar]:
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


class ChainedEncoder(BaseEncoder, Sequence[EncoderVar]):
    r"""Represents function composition of encoders."""

    requires_fit: bool = True  # type: ignore[misc]

    encoders: list[EncoderVar]
    r"""List of encoders."""

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

    def __init__(self, *encoders: EncoderVar, simplify: bool = True) -> None:
        super().__init__()

        self.encoders = []

        for encoder in encoders:
            if simplify and isinstance(encoder, ChainedEncoder):
                for enc in encoder:
                    self.encoders.append(enc)
            else:
                self.encoders.append(encoder)
        self.requires_fit = any(e.requires_fit for e in self.encoders)

    def __len__(self) -> int:
        r"""Return number of chained encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> EncoderVar:
        ...

    @overload
    def __getitem__(self, index: slice) -> ChainedEncoder[EncoderVar]:
        ...

    def __getitem__(self, index):
        r"""Get the encoder at the given index."""
        if isinstance(index, int):
            return self.encoders[index]
        if isinstance(index, slice):
            return ChainedEncoder(*self.encoders[index])
        raise ValueError(f"Index {index} not supported.")

    def fit(self, data: Any, /) -> None:
        for encoder in reversed(self.encoders):
            encoder.fit(data)
            data = encoder.encode(data)

    def encode(self, data: Any, /) -> Any:
        for encoder in reversed(self.encoders):
            data = encoder.encode(data)
        return data

    def decode(self, data: Any, /) -> Any:
        for encoder in self.encoders:
            data = encoder.decode(data)
        return data


class MappingEncoder(BaseEncoder, Mapping[KeyVar, EncoderVar]):
    r"""Encoder that maps keys to encoders."""

    requires_fit: bool = True  # type: ignore[misc]

    encoders: Mapping[KeyVar, EncoderVar]
    r"""Mapping of keys to encoders."""

    def __init__(self, encoders: Mapping[KeyVar, EncoderVar]) -> None:
        super().__init__()
        self.encoders = encoders
        self.requires_fit = any(e.requires_fit for e in self.encoders.values())

    @overload
    def __getitem__(self, key: KeyVar) -> EncoderVar:
        ...

    @overload
    def __getitem__(self, key: list[KeyVar]) -> MappingEncoder[KeyVar, EncoderVar]:
        ...

    def __getitem__(
        self, key: KeyVar | list[KeyVar]
    ) -> EncoderVar | MappingEncoder[KeyVar, EncoderVar]:
        r"""Get the encoder for the given key."""
        if isinstance(key, list):
            return MappingEncoder({k: self.encoders[k] for k in key})
        return self.encoders[key]

    def __len__(self) -> int:
        return len(self.encoders)

    def __iter__(self) -> Iterator[KeyVar]:
        return iter(self.encoders)

    def fit(self, data: Mapping[KeyVar, Any], /) -> None:
        assert set(data.keys()) == set(self.encoders.keys())
        for key in data:
            self.encoders[key].fit(data[key])

    def encode(self, data: Mapping[KeyVar, Any], /) -> Mapping[KeyVar, Any]:
        return {
            key: encoder.encode(data[key]) for key, encoder in self.encoders.items()
        }

    def decode(self, data: Mapping[KeyVar, Any], /) -> Mapping[KeyVar, Any]:
        return {
            key: encoder.decode(data[key]) for key, encoder in self.encoders.items()
        }


class DuplicateEncoder(BaseEncoder):
    r"""Duplicate encoder multiple times (references same object)."""

    def __init__(self, encoder: BaseEncoder, n: int = 1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(self.base_encoder for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: Any, /) -> None:
        return self.encoder.fit(data)

    def encode(self, data, /):
        return self.encoder.encode(data)

    def decode(self, data, /):
        return self.encoder.decode(data)


class CloneEncoder(BaseEncoder):
    r"""Clone encoder multiple times (distinct copies)."""

    def __init__(self, encoder: BaseEncoder, n: int = 1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(deepcopy(self.base_encoder) for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def fit(self, data: Any, /) -> None:
        return self.encoder.fit(data)

    def encode(self, data, /):
        return self.encoder.encode(data)

    def decode(self, data, /):
        return self.encoder.decode(data)
