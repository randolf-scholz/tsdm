r"""Protocols for encoders and decoders."""

__all__ = [
    "Reduction",
    "Expansion",
    "SupportsEncode",
    "SupportsDecode",
    "SupportsFit",
    "SupportSimplify",
    "SupportsParameters",
    "SupportsSerialization",
]

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, Self, runtime_checkable

from tsdm.types.aliases import FilePath


class Reduction[Xs: tuple, Y](Protocol):
    r"""Protocol for objects that support reduction."""

    def __call__(self, xs: Xs, /) -> Y: ...


class Expansion[X, Ys: tuple](Protocol):
    r"""Protocol for objects that support expansion."""

    def __call__(self, x: X, /) -> Ys: ...


@runtime_checkable
class SupportsEncode[X, Y](Protocol):
    r"""Protocol for objects that support encoding."""

    def encode(self, x: X, /) -> Y: ...


@runtime_checkable
class SupportsDecode[X, Y](Protocol):
    r"""Protocol for objects that support decoding."""

    def decode(self, y: Y, /) -> X: ...


@runtime_checkable
class SupportsFit[X](Protocol):
    r"""Protocol for objects that support fitting."""

    def fit(self, x: X, /) -> None: ...


@runtime_checkable
class SupportSimplify(Protocol):  # Encoder[X, Y]
    r"""Protocol for objects that support simplification."""

    def simplify(self) -> Any: ...


@runtime_checkable
class SupportsParameters(Protocol):
    r"""Protocol for objects that support parameters."""

    @property
    def params(self) -> Mapping[str, Any]: ...
    def validate_params(self) -> None: ...


@runtime_checkable
class SupportsSerialization(Protocol):
    r"""Protocol for serializable encoders."""

    @abstractmethod
    def serialize(self, filepath: FilePath, /) -> None: ...

    @classmethod
    @abstractmethod
    def deserialize(cls, filepath: FilePath, /) -> Self: ...
