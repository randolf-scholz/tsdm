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

from typing import Protocol, runtime_checkable


@runtime_checkable
class Array(Protocol):
    r"""We just test for shape, since e.g. tf.Tensor does not have ndim."""

    @property
    def shape(self) -> tuple[int, ...] | list[int]:
        r"""Shape of the array."""
        return ()

    # @property
    # def dtype(self) -> Any:
    #     ...


@runtime_checkable
class NTuple(Protocol):
    r"""To check for namedtuple."""

    @property
    def _fields(self) -> tuple[str, ...]:
        return ()

    def _asdict(self) -> dict[str, object]:
        ...
