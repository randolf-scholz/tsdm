r"""Encoder that serializes as CSV file."""

__all__ = ["CSVEncoder"]

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from tsdm.constants import EMPTY_MAP
from tsdm.encoders.base import StaticEncoder
from tsdm.pprint import pprint_repr
from tsdm.types.aliases import FilePath


@pprint_repr
@dataclass(init=False, slots=True)
class CSVEncoder(StaticEncoder[pd.DataFrame, Path]):
    r"""Encode the data into a CSV file."""

    DEFAULT_READ_OPTIONS: ClassVar[dict] = {}
    DEFAULT_WRITE_OPTIONS: ClassVar[dict] = {"index": False}

    path_generator: Callable[[pd.DataFrame], Path]
    r"""The generates the name for the CSV file."""

    _: KW_ONLY

    csv_read_options: dict[str, Any]
    r"""The options for the read_csv function."""
    csv_write_options: dict[str, Any]
    r"""The options for the to_csv function."""

    def __init__(
        self,
        filename_or_generator: FilePath | Callable[[pd.DataFrame], Path],
        *,
        csv_write_options: Mapping[str, Any] = EMPTY_MAP,
        csv_read_options: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        self.csv_read_options = self.DEFAULT_READ_OPTIONS | dict(csv_read_options)
        self.csv_write_options = self.DEFAULT_WRITE_OPTIONS | dict(csv_write_options)
        self.path_generator = (
            filename_or_generator
            if callable(filename_or_generator)
            else lambda _: Path(filename_or_generator)
        )

    def encode(self, data: pd.DataFrame, /) -> Path:
        path = self.path_generator(data)
        data.to_csv(path, **self.csv_write_options)
        return path

    def decode(self, str_or_path: Path, /) -> pd.DataFrame:
        return pd.read_csv(str_or_path, **self.csv_read_options)
