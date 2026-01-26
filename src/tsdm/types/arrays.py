r"""Array protocol types for the numerical backend.

Here is a summary, based on the library versions:

- `numpy`:
- `pandas`:
- `polars`:
- `torch`:
- `pyarrow`:

|                             | np.ndarray | torch.Tensor | pd.Index | pd.EA | pd.Series       | pl.Series | pa.Array | pd.DataFrame | pl.DataFrame | pa.Table |
|-----------------------------|------------|--------------|----------|-------|-----------------|-----------|----------|--------------|--------------|----------|
| dimensionality              | N          | N            | 1        | 1     | 1               | 1         | 1        | 2            | 2            | 2        |
| comparisons                 | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ✅            | ✅            | ❌        |
| arithmetic                  | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ✅            | ❌            | ❌        |
| mutation                    | ✅          | ✅            | ❌        | ❌     | ✅               | ❌         | ❌        | ✅            | ❌            | ❌        |
| `__array__`                 | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ✅        | ✅            | ✅            | ✅        |
| `__array_ufunc__`           | ✅          | ❌            | ✅        | ✅     | ✅               | ❌         | ❌        | ✅            | ❌            | ❌        |
| `__dataframe__`             | ❌          | ❌            | ❌        | ❌     | ❌               | ❌         | ❌        | ✅            | ✅            | ✅        |
| `.shape`                    | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ✅            | ✅            | ✅        |
| `.dtype`                    | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ❌        | ❌            | ❌            | ❌        |
| `.ndim`                     | ✅          | ✅            | ✅        | ✅     | ✅               | ❌         | ❌        | ✅            | ❌            | ❌        |
| `.device`                   | ✅          | ✅            | ❌        | ❌     | ❌               | ❌         | ❌        | ❌            | ❌            | ❌        |
| `item()`                    | ✅          | ✅            | ✅        | ❌     | ✅               | ✅         | ❌        | ❌            | ✅            | ❌        |
| `__matmul__()`              | ✅          | ✅            | ❌        | ❌     | ✅               | ✅         | ❌        | ✅            | ❌            | ❌        |
| `__len__()`                 | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ✅        | ✅            | ✅            | ✅        |
| `__iter__()`                | ✅          | ✅            | ✅        | ✅     | ✅               | ✅         | ✅        | ✅            | ✅            | ❌        |
| `iter()`                    | ROW        | ROW          | ROW      | ROW   | ROW             | ROW       | ROW      | COL NAME     | COL          | COL      |
| `__getitem__(index)`        | ROW        | ROW          | ROW      | ROW   | ROWS (⚡UNSAFE⚡) | ROW       | ROW      | ❌            | ROW          | COL      |
| `__getitem__(list[index])`  | ROWS       | ROWS         | ROWS     | ROWS  | ROWS (⚡UNSAFE⚡) | ROWS      | ❌        | COLS         | ROWS         | ❌        |
| `__getitem__(slice[index])` | ROWS       | ROWS         | ROWS     | ROWS  | ROWS            | ROWS      | ROWS     | ROWS         | ROWS         | ROWS     |
| `__getitem__(label)`        | ❌          | ❌            | ❌        | ❌     | ROW             | ❌         | ❌        | COL          | COL          | COL      |
| `__getitem__(list[label])`  | ❌          | ❌            | ❌        | ❌     | ROWS            | ❌         | ❌        | COLS         | COLS         | ❌        |
| `__getitem__(slice[label])` | ❌          | ❌            | ❌        | ❌     | ROWS            | ❌         | ❌        | ROWS         | COLS         | ❌        |
| `__getitem__(list[bool])`   | ROWS       | ROWS         | ROWS     | ROWS  | ROWS            | ❌         | ❌        | ROWS         | ❌            | ❌        |

Note:
    - The summary shows that `pyarrow` objects lack support for many operations.
    - mutability means that `x += 1` changes `x` in-place, resulting in object with identical id.
      Note that for instance `pl.Series` still supports `+=` but it does actually create a new object.

From this table, we are inclined to derive several protocols:

- `Table`-like protocol for 2d column-oriented arrays pd.DataFrame, pl.DataFrame and pa.Table.

Warning:
    `pandas.Series` has inherently usafe indexing! This is because `series[int]` and `series[list[int]]`

    - return the same as `series.iloc[int]` and `series.iloc[list[int]]` if series not indexed by integers.
    - return the same as `series.loc[int]` and `series.loc[list[int]]` if the series is indexed by integers.

    `polars` and `pyarrow` are strict by demanding that rows are indexed by indices, and columns by labels (strings).
"""  # noqa: E501, W505

__all__ = [
    "ArrayLike",
    "SeriesLike",
    "TableLike",
]

from collections.abc import Iterator
from typing import Protocol, Self, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ArrayLike[V](Protocol):
    r"""An n-dimensional array of a single homogeneous data type.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Series`
        - `pandas.extensions.ExtensionArray`
        - `polars.DataFrame`
        - `polars.Series`
        - `pyarrow.Array`
        - `pyarrow.Table`
        - `torch.Tensor`

    Note:
        `pyarrow` does not support element-wise comparisons.

    References:
        - https://docs.python.org/3/c-api/buffer.html
        - https://numpy.org/doc/stable/reference/arrays.interface.html
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...


@runtime_checkable
class SeriesLike[V](Protocol):
    r"""A 1d-array of homogeneous data type.

    Examples:
        - `pandas.Index`
        - `pandas.Series`
        - `polars.Series`
        - `pandas.extensions.ExtensionArray`
        - `pyarrow.Array`

    Counter-Examples:
        - `numpy.ndarray`     lacks `equals`
        - `pandas.DataFrame`  lacks `equals`
        - `polars.DataFrame`  lacks `equals`
        - `pyarrow.Table`     lacks `equals`
        - `torch.Tensor`      lacks `equals`

    NOTE: Many methods have subtle differences between backends:
     - `diff`: gives discrete differences for polars and pandas, but not for pyarrow
     - `value_counts`: polars returns a DataFrame, pandas a Series, pyarrow a StructArray
     - `unique`: polars and pyarrow return `Self`, pandas returns `np.ndarray` or ExtensionArray.
     - `to_numpy`: is superfluous.

    References:
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[V]: ...
    @overload
    def __getitem__(self, key: int, /) -> V: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...

    def equals(self, other: Self, /) -> bool:
        r"""Check if the series is equal to another series."""
        ...


@runtime_checkable
class TableLike(Protocol):
    r"""A 2d column-oriented array with heterogenous data types.

    That it, it is a column-oriented 2d tensor which allows heterogenous data types.

    Note:
        In contrast to tensors (row-oriented), tables are column-oriented. Therefore,
        `__getitem__` returns a column, which is a SeriesKind, i.e. homogeneous 1d tensor.

    Note: The following methods differ between backends:
        - `pyarrow` does not support element-wise comparisons.
        - `iter`: yields columns for polars and pandas, but rows for pyarrow
             This is because `pyarrow` does not actually define `__iter__`.
        - `take`: not supported by polars
        - `columns`: pandas and polars return column names, pyarrow returns list of Arrays
        - `drop`: polars currently doing signature change
        - `filter`: pandas goes over columns, polars over rows

    Examples:
        - `pandas.DataFrame`
        - `polars.DataFrame`
        - `pyarrow.Table`

    Counter-Examples:
        - `numpy.ndarray`  lacks `__dataframe__`
        - `pandas.Series`  lacks `__dataframe__`
        - `pandas.extensions.ExtensionArray`  lacks `__dataframe__`
        - `polars.Series`  lacks `__dataframe__`
        - `pyarrow.Array`  lacks `__dataframe__`
        - `torch.Tensor`  lacks `__dataframe__`

    References:
        - https://numpy.org/devdocs/user/basics.interoperability.html
        - https://data-apis.org/dataframe-protocol/latest/index.html
    """

    @property
    def shape(self) -> tuple[int, int]: ...

    def __array__(self) -> NDArray[np.object_]: ...
    def __dataframe__(self, *, allow_copy: bool = True) -> object: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: str, /) -> SeriesLike: ...  # yields a column

    def equals(self, other: Self, /) -> bool:
        r"""Check if the table is equal to another table."""
        ...
