r"""Compare Torch NaN reductions with NumPy."""

from warnings import catch_warnings

import numpy as np
import pytest
import torch

from tsdm.backend import torch as backend
from tsdm.types.aliases import Axis


@pytest.mark.parametrize("name", ["nanmin", "nanmax", "nanstd"])
@pytest.mark.parametrize("axis", [None, 0, -1, (0, 1), ()])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[np.nan, np.nan], [1.0, 3.0]],
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[np.nan, -np.inf], [np.inf, 1.0]],
        [[np.nan, np.inf], [np.nan, np.inf]],
    ],
)
def test_nan_reductions(
    *, name: str, axis: Axis, keepdims: bool, data: list[list[float]]
) -> None:
    r"""Match values, shapes, and dtypes, including valid infinities."""
    array = np.asarray(data)
    with (
        catch_warnings(action="ignore", category=RuntimeWarning),
        np.errstate(invalid="ignore"),
    ):
        expected = getattr(np, name)(array, axis=axis, keepdims=keepdims)
    actual = getattr(backend, name)(
        torch.from_numpy(array), axis=axis, keepdims=keepdims
    ).numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("axis", [None, 0, -1, (0, 1), ()])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("kind", ["empty", "complex"])
def test_nanstd_special_inputs(*, axis: Axis, keepdims: bool, kind: str) -> None:
    r"""Match NumPy for empty slices and complex deviations."""
    array = (
        np.empty((2, 0))
        if kind == "empty"
        else np.array([[1 + 2j, np.nan], [3 - 4j, 5 + 6j]])
    )
    with catch_warnings(action="ignore", category=RuntimeWarning):
        expected = np.nanstd(array, axis=axis, keepdims=keepdims)
    actual = backend.nanstd(
        torch.from_numpy(array), axis=axis, keepdims=keepdims
    ).numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
