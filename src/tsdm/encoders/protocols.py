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
    "SupportsBackend",
]

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, Self, runtime_checkable

from tsdm.backend import Backend, get_backend
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


class SupportsBackend[X, Y](Protocol):
    r"""Encoder equipped with a backend."""

    backend: Backend

    def set_backend_from_data(self, x: X, /) -> None:
        self.backend = get_backend(x)

    def switch_backend(self, backend: str, /) -> None:
        r"""Switch the backend of the encoder."""
        self.backend = Backend(backend)

        # recast the parameters
        self.recast_parameters()

    def recast_parameters(self) -> None:
        r"""Recast the parameters to the current backend."""
        raise NotImplementedError

    # pre_fit_hooks: ClassVar[list[Fn]] = [set_backend_from_data]
