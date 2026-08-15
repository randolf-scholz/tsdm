__all__ = ["assert_arrays_equal", "assert_arrays_close"]


from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from polars import testing as pl_testing


def assert_arrays_equal[T: Any](array: T, reference: T, /) -> None:
    r"""Assert that the arrays are equal."""
    if type(array) is not type(reference):
        raise AssertionError(f"{type(array)=} != {type(reference)=}")

    match array:
        case pd.Series():
            pd.testing.assert_series_equal(array, reference)
        case pd.Index():
            pd.testing.assert_index_equal(array, reference)
        case pd.DataFrame():
            pd.testing.assert_frame_equal(array, reference)
        case np.ndarray():
            np.testing.assert_array_equal(array, reference)
        case pl.Series():
            pl_testing.assert_series_equal(array, reference)
        case pl.DataFrame():
            pl_testing.assert_frame_equal(array, reference)
        case torch.Tensor():
            torch.testing.assert_close(array, reference, rtol=0, atol=0)
        case Sequence() as seq:
            if len(array) != len(reference):
                raise AssertionError(f"{len(array)=} != {len(reference)=}")
            if any(a != b for a, b in zip(seq, reference, strict=True)):
                raise AssertionError(f"{array=} != {reference=}")
        case _:
            raise TypeError(f"Unsupported {type(array)=}")


def assert_arrays_close[T: Any](
    array: T,
    reference: T,
    /,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    r"""Assert that the arrays are close within tolerance."""
    if type(array) is not type(reference):
        raise AssertionError(f"{type(array)=} != {type(reference)=}")

    match array:
        case pd.Series():
            pd.testing.assert_series_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case pd.Index():
            pd.testing.assert_index_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case pd.DataFrame():
            pd.testing.assert_frame_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case np.ndarray():
            np.testing.assert_allclose(array, reference, atol=atol, rtol=rtol)
        case pl.Series():
            pl_testing.assert_series_equal(
                array, reference, check_exact=False, abs_tol=atol, rel_tol=rtol
            )
        case pl.DataFrame():
            pl_testing.assert_frame_equal(
                array, reference, check_exact=False, abs_tol=atol, rel_tol=rtol
            )
        case torch.Tensor() as tensor:
            torch.testing.assert_close(tensor, reference, atol=atol, rtol=rtol)
        case Sequence() as seq:
            if len(array) != len(reference):
                raise AssertionError(f"{len(array)=} != {len(reference)=}")
            if any(
                abs(a - b) > atol + rtol * abs(b)
                for a, b in zip(seq, reference, strict=True)
            ):
                raise AssertionError(f"{array=} != {reference=}")
        case _:
            raise TypeError(f"Unsupported {type(array)=}")
