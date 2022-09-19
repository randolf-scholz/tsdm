r"""Protocols used in tsdm.

References
----------
- https://www.python.org/dev/peps/pep-0544/
- https://docs.python.org/3/library/typing.html#typing.Protocol
- https://numpy.org/doc/stable/reference/c-api/array.html
"""

__all__ = [
    # Classes
    "Array",
    "NTuple",
]

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Array(Protocol):
    r"""We just test for shape, since e.g. tf.Tensor does not have ndim."""

    @property
    def shape(self) -> Iterable[int]:
        r"""Yield the shape of the array."""
        ...

    @property
    def dtype(self) -> Any:
        r"""Yield the data type of the array."""
        ...


@runtime_checkable
class NTuple(Protocol):
    r"""To check for namedtuple."""

    @property
    def _fields(self) -> tuple[str, ...]:
        return ()

    def _asdict(self) -> dict[str, object]:
        ...
