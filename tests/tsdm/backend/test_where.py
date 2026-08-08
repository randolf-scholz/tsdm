r"""Test the pandas backend ``where`` implementation."""

import numpy as np
import pandas as pd
import pytest

from tsdm.backend.pandas import where


@pytest.mark.parametrize("container", [pd.DataFrame, pd.Series, pd.Index])
@pytest.mark.parametrize(
    "array_arguments",
    [(), ("cond", "a"), ("cond", "b"), ("a", "b")],
)
def test_where_propagates_nulls_from_selected_argument(
    container: type[pd.DataFrame | pd.Series | pd.Index],
    array_arguments: tuple[str, ...],
) -> None:
    r"""Check that nulls propagate from the branch selected by ``cond``."""
    rng = np.random.default_rng(0)
    shape = (3, 4) if container is pd.DataFrame else (12,)
    cond_values = rng.integers(0, 2, size=shape, dtype=bool)
    cond_missing = rng.random(size=shape) < 0.25
    cond_data = cond_values.astype(object)
    cond_data[cond_missing] = pd.NA
    a_data = rng.normal(size=shape)
    a_data[rng.random(size=shape) < 0.25] = np.nan
    b_data = rng.normal(size=shape)
    b_data[rng.random(size=shape) < 0.25] = np.nan

    operands = {
        "cond": container(cond_data),
        "a": container(a_data),
        "b": container(b_data),
    }
    operands.update({name: np.asarray(operands[name]) for name in array_arguments})

    result = where(operands["cond"], operands["a"], operands["b"])
    expected = np.where(cond_values & ~cond_missing, np.isnan(a_data), np.isnan(b_data))
    np.testing.assert_array_equal(pd.isna(result), expected)
