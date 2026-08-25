r"""Encoder for column-wise Polars DataFrame transforms."""

__all__ = ["FrameEncoder"]

from dataclasses import field

import polars as pl

from tsdm.constants import UNDEFINED
from tsdm.encoders.base import Encoder, EncoderDict
from tsdm.pprint import pprint_mapping


@pprint_mapping
# FIXME: https://github.com/python/cpython/issues/140596
class FrameEncoder[K: str](
    EncoderDict[
        pl.DataFrame,
        pl.DataFrame,
        K,
        Encoder[pl.Series, pl.Series],
    ]
):
    r"""Encode a Polars DataFrame by column-wise transformations.

    Each configured encoder receives the corresponding column as a Polars
    :class:`Series`. All columns without an encoder pass through unchanged. The
    fitted column dtypes are restored when decoding.

    Args:
        encoders: A mapping from column names to their encoders.

    Examples:
        >>> import polars as pl
        >>> from polars.testing import assert_frame_equal
        >>> from tsdm.encoders import wrap
        >>> frame = pl.DataFrame({"id": [10, 20], "value": [1, 2], "label": ["a", "b"]})
        >>> encoder = FrameEncoder({"value": wrap(lambda x: x + 1, lambda x: x - 1)})
        >>> encoder.fit(frame)
        >>> encoded = encoder.encode(frame)
        >>> encoded.to_dict(as_series=False)
        {'id': [10, 20], 'value': [2, 3], 'label': ['a', 'b']}
        >>> assert_frame_equal(frame, encoder.decode(encoded))

    Todo:
        We want encoding groups, so for example, applying an encoder to a group
        of columns.

        - [ ] Add support for groups of column-encoders

    See Also:
        Similar to :class:`sklearn.compose.ColumnTransformer`.
    """

    # fitted attributes
    original_schema: pl.Schema = field(init=False, default=UNDEFINED)

    def fit(self, data: pl.DataFrame, /) -> None:
        data = data.clone()
        self.original_schema = data.schema

        errors: list[Exception] = []
        typ = type(self).__name__
        for column, encoder in self.encoders.items():
            try:
                encoder.fit(data[column])
            except Exception as exc:
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{column}]: Failed to fit {enc}.")
                errors.append(exc)

        if errors:
            raise ExceptionGroup(f"{typ}: Failed to fit column encoders.", errors)

    def encode(self, data: pl.DataFrame, /) -> pl.DataFrame:
        for column, encoder in self.encoders.items():
            encoded = encoder.encode(data[column]).rename(column)
            data = data.with_columns(encoded)

        return data

    def decode(self, data: pl.DataFrame, /) -> pl.DataFrame:
        for column, encoder in self.encoders.items():
            decoded = encoder.decode(data[column]).rename(column)
            data = data.with_columns(decoded)

        schema = pl.Schema(
            {column: self.original_schema[column] for column in data.columns}
        )
        return data.cast(schema)
