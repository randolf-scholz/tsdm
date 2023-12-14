r"""Tests for TableEncoders."""

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

angle = np.linspace(0, np.pi, 5)

sample_data = {
    # discrete
    "col1": [1, 2, 3, 4, 5],
    # categorical
    "col2": ["cat", "dog", "dog", "dog", "cat"],
    # angle
    "col_x": np.cos(angle),
    "col_y": np.sin(angle),
}


pandas_table = pd.DataFrame(sample_data)
polars_table = pl.DataFrame(sample_data)
arrow_table = pa.Table.from_pydict(sample_data)

TABLES = {
    "pandas": pandas_table,
    "polars": polars_table,
    "arrow": arrow_table,
}


def test_one_to_one():
    """Test an encoder that transforms a single column to a single column."""


def test_one_to_many():
    """Test an encoder that transforms a single column to multiple columns."""


def test_many_to_many():
    """Test an encoder that transforms multiple columns to multiple columns."""


def test_many_to_one():
    """Test an encoder that transforms multiple columns to a single column."""
