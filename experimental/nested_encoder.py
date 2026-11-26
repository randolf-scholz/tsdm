r"""Experimental encoders."""

__all__ = [
    "NDArrayToTensor",
    "NestedEncoder",
    "nest_encoder",
    "NestedBuiltin",
    "recurse_on_nested_builtin",
]

from collections.abc import Callable as Fn
from dataclasses import KW_ONLY, dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from tsdm.encoders.base import Encoder, FittableEncoder, StaticEncoder, invert, simplify
from tsdm.utils import pretty_repr

type NestedBuiltin[T] = (
    T
    | tuple[T, ...]  # leaf-tuple
    | tuple[NestedBuiltin[T], ...]
    | set[T]  # leaf-set
    | set[NestedBuiltin[T]]
    | frozenset[T]  # leaf-frozenset
    | frozenset[NestedBuiltin[T]]
    | list[T]  # leaf-list
    | list[NestedBuiltin[T]]
    | dict[str, T]  # leaf-dict
    | dict[str, NestedBuiltin[T]]
)
r"""Type alias for leaves nested in tuples, sets, frozensets, lists, or dictionaries."""


def recurse_on_nested_builtin[T, R](
    arg: NestedBuiltin[T],
    /,
    *,
    leaf_fn: Fn[[T], R],
    leaf_type: type[T],
    # optional arguments
    leaf_priotizied: bool = False,
) -> NestedBuiltin[R]:
    r"""Recursively apply a function to the leaves of a nested data structure.

    Args:
        arg: Nested data structure to apply the function to.
        leaf_fn: Function to apply to the leaf nodes.
        leaf_type: Type of the leaf nodes.
        leaf_priotizied: Whether to check the leaf type before container types.
            If false, supported containers are traversed even if they match
            `leaf_type`.

    Returns:
        A structure with the same container types and transformed leaf values.

    Raises:
        TypeError: If a value is neither a supported builtin container nor a leaf.
    """

    def recurse(obj: NestedBuiltin[T], /) -> NestedBuiltin[R]:
        match obj:
            case leaf if leaf_priotizied and isinstance(leaf, leaf_type):
                return leaf_fn(leaf)
            case dict(mapping):
                return {k: recurse(v) for k, v in mapping.items()}
            case list(seq):
                return [recurse(obj) for obj in seq]
            case tuple(seq):
                return tuple(recurse(obj) for obj in seq)
            case frozenset(items):
                return frozenset(recurse(obj) for obj in items)
            case set(items):
                # FIXME: https://github.com/python/typeshed/issues/9571
                return {recurse(obj) for obj in items}  # pyright: ignore[reportUnhashable]
            case leaf if isinstance(leaf, leaf_type):
                return leaf_fn(leaf)
            case _:
                raise TypeError(
                    f"Unsupported type: {type(obj)} not an instance of {leaf_type} or one of the builtin containers"
                    f" {dict, list, tuple, frozenset, set}."
                )

    return recurse(arg)


@pretty_repr
@dataclass(slots=True)
class NestedEncoder[X, Y](FittableEncoder[NestedBuiltin[X], NestedBuiltin[Y]]):
    r"""Apply an encoder recursively to nested data structure.

    Any instances of the leaf type will be encoded using the encoder.
    Containers in the standard library will be recursed into
    (applies to `list`, `tuple`, `dict`, `set` and `frozenset`).
    Other types will raise `TypeError`.

    TODO: add support to pass other types as-is.
    """

    encoder: Encoder[X, Y]
    r"""The encoder to apply nested."""

    _: KW_ONLY

    leaf_type: type[X]
    r"""The type of the leaf elements."""
    output_leaf_type: type[Y]
    r"""The type of the output elements."""

    # FIXME: https://github.com/python/typing/issues/548
    def __invert__(self) -> NestedEncoder[Y, X]:
        return nest_encoder(
            invert(self.encoder),
            leaf_type=self.output_leaf_type,
            output_leaf_type=self.leaf_type,
        )

    def fit(self, x: NestedBuiltin[X], /) -> None:
        pass

    def encode(self, x: NestedBuiltin[X], /) -> NestedBuiltin[Y]:
        return recurse_on_nested_builtin(
            x,
            leaf_type=self.leaf_type,
            leaf_fn=self.encoder.encode,
        )

    def decode(self, y: NestedBuiltin[Y], /) -> NestedBuiltin[X]:
        return recurse_on_nested_builtin(
            y,
            leaf_type=self.output_leaf_type,
            leaf_fn=self.encoder.decode,
        )

    def simplify(self) -> NestedEncoder[X, Y]:
        return NestedEncoder(
            simplify(self.encoder),
            leaf_type=self.leaf_type,
            output_leaf_type=self.output_leaf_type,
        )


def nest_encoder[X, Y](
    encoder: Encoder[X, Y], /, *, leaf_type: type[X], output_leaf_type: type[Y]
) -> NestedEncoder[X, Y]:
    r"""Create a nested encoder that applies the given encoder recursively."""
    return NestedEncoder(
        encoder, leaf_type=leaf_type, output_leaf_type=output_leaf_type
    )


class NDArrayToTensor(StaticEncoder[NestedBuiltin[NDArray], NestedBuiltin[Tensor]]):
    r"""Encodes nested data as tensors."""

    def encode(self, x: NestedBuiltin[NDArray], /) -> NestedBuiltin[Tensor]:
        return recurse_on_nested_builtin(x, leaf_fn=torch.tensor, leaf_type=np.ndarray)

    def decode(self, y: NestedBuiltin[Tensor], /) -> NestedBuiltin[NDArray]:
        return recurse_on_nested_builtin(y, leaf_fn=Tensor.numpy, leaf_type=Tensor)  # pyright: ignore[reportArgumentType]
