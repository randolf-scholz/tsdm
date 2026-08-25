r"""Encoder for columns-wise transforms."""

__all__ = ["FrameEncoder"]

from dataclasses import field

import pandas as pd

from tsdm.constants import UNDEFINED
from tsdm.encoders.base import Encoder, EncoderDict
from tsdm.pprint import pprint_mapping


@pprint_mapping
# FIXME: https://github.com/python/cpython/issues/140596
class FrameEncoder[K](EncoderDict[pd.DataFrame, pd.DataFrame, K, Encoder]):
    r"""Encode a DataFrame by group-wise transformations.

    The DataFrame index is reset before fitting, encoding, and decoding, so encoders can target
    both data columns and named index levels. All fields not assigned an encoder pass through
    unchanged. The original index and dtypes are restored when decoding.

    Args:
        encoders: A mapping from column or named index-level labels to their encoders.

    Examples:
        >>> from pandas import DataFrame
        >>> from pandas.testing import assert_frame_equal
        >>> from tsdm.encoders import wrap
        >>> frame = DataFrame({"id": [10, 20], "value": [1, 2], "label": ["a", "b"]})
        >>> frame = frame.set_index("id")
        >>> encoder = FrameEncoder({"value": wrap(lambda x: x + 1, lambda x: x - 1)})
        >>> encoder.fit(frame)
        >>> encoded = encoder.encode(frame)
        >>> encoded.to_dict("list")
        {'value': [2, 3], 'label': ['a', 'b']}
        >>> assert_frame_equal(frame, encoder.decode(encoded))

    Note:
        `...` (`Ellipsis`) is treated as an ordinary mapping key, not as a wildcard for unassigned
        columns. Thus, it only targets a column literally labelled `...`; otherwise, fitting or
        transforming raises `KeyError`.

    Todo:
        We want encoding groups, so for example, applying an encoder to a group of columns.

        - [ ] Add support for groups of column-encoders

    See Also:
        Similar to `sklearn.compose.ColumnTransformer`.
    """

    # fitted attributes
    original_index: pd.Index = field(init=False, default=UNDEFINED)
    original_schema: pd.Series = field(init=False, default=UNDEFINED)

    def fit(self, data: pd.DataFrame, /) -> None:
        data = data.copy(deep=True)
        index = data.index.to_frame()
        self.original_index = index.columns

        data = data.reset_index()
        self.original_schema = data.dtypes

        errors: list[Exception] = []
        typ = type(self).__name__
        for group, encoder in self.encoders.items():
            try:
                encoder.fit(data[group])
            except Exception as exc:
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{group}]: Failed to fit {enc}.")
                errors.append(exc)

        if errors:
            raise ExceptionGroup(f"{typ}: Failed to fit column encoders.", errors)

    def encode(self, data: pd.DataFrame, /) -> pd.DataFrame:
        data = data.reset_index()

        for group, encoder in self.encoders.items():
            data[group] = encoder.encode(data[group])

        index_columns = data.columns.intersection(self.original_index)
        data = data.set_index(index_columns.tolist())
        return data

    def decode(self, data: pd.DataFrame, /) -> pd.DataFrame:
        data = data.reset_index()

        for group, encoder in self.encoders.items():
            data[group] = encoder.decode(data[group])

        # Restore index order + dtypes
        data = data.astype(self.original_schema[data.columns])
        index_columns = data.columns.intersection(self.original_index)
        data = data.set_index(index_columns.tolist())
        return data
