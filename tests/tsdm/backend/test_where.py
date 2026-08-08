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
    pd.testing.assert_frame_equal(
        pd.DataFrame(pd.isna(result)),
        pd.DataFrame(expected),
    )


def test_where_preserves_string_dataframe_metadata() -> None:
    r"""Check that a boolean condition container need not store branch values."""
    index = pd.Index(["first", "second"], name="row")
    columns = pd.Index(["left", "right"], name="side")
    cond = pd.DataFrame(
        [[True, pd.NA], [False, True]], index=index, columns=columns, dtype="boolean"
    )
    a = pd.DataFrame(
        [["a00", "a01"], ["a10", "a11"]],
        index=index,
        columns=columns,
        dtype=pd.StringDtype(),
    )
    b = pd.DataFrame(
        [["b00", "b01"], ["b10", "b11"]],
        index=index,
        columns=columns,
        dtype=pd.StringDtype(),
    )
    expected = pd.DataFrame(
        [["a00", "b01"], ["b10", "a11"]],
        index=index,
        columns=columns,
        dtype=pd.StringDtype(),
    )

    result = where(cond, a, b)

    pd.testing.assert_frame_equal(result, expected)
