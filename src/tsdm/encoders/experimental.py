r"""Experimental encoders."""

__all__ = ["NestedEncoder", "nest_encoder"]

from dataclasses import KW_ONLY, dataclass

from tsdm.types.aliases import NestedBuiltin
from tsdm.utils.decorators import pprint_repr
from tsdm.utils.funcutils import recurse_on_nested_builtin

from .base import Encoder, FittableEncoder, invert, simplify


@pprint_repr
@dataclass
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
