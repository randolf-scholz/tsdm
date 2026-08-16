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

    Similar to `sklearn.compose.ColumnTransformer`.

    Per-column encoding is possible through the dictionary input.
    In this case, the positions of the columns in the encoded DataFrame should coincide with the
    positions of the columns in the input DataFrame.

    Todo: We want encoding groups, so for example, applying an encoder to a group of columns.

    - [ ] Add support for groups of column-encoders
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

        # fit the encoders one by one
        for group, encoder in self.encoders.items():
            try:
                encoder.fit(data[group])
            except Exception as exc:
                typ = type(self).__name__
                enc = type(encoder).__name__
                exc.add_note(f"{typ}[{group}]: Failed to fit {enc}.")
                raise

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
