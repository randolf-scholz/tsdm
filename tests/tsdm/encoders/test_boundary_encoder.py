r"""Tests for Boundary Encoders."""

import numpy as np
import pandas as pd
import pytest
import torch

from tsdm.encoders import BoundaryEncoder
from tsdm.types.arrays import NumericalTensor

DATA_1D = [
    float("-inf"),
    -1.1,
    -1.0,
    -0.9,
    -0.0,
    +0.0,
    +0.3,
    +1.0,
    +1.5,
    float("+inf"),
]
DATA_2D = [
    [-2.0, -1.1, -1.0, -0.9],
    [ 0.0,  0.3,  0.5,  1.0],
    [ 1.5,  2.0,  2.5,  3.0],
]  # fmt: skip

BOUNDS: list[tuple[float | None, float | None]] = [
    (-1, +1),
    (0, 1),
    (0, float("inf")),
    (0, None),
    (0, float("nan")),
    (0, pd.NA),
]
TENSORS: dict[str, NumericalTensor[float]] = {
    "numpy-1D"             : np.array(DATA_1D),
    "numpy-2D"             : np.array(DATA_2D),
    "torch-1D"             : torch.tensor(DATA_1D),
    "torch-2D"             : torch.tensor(DATA_2D),
    "pandas[numpy]-index"  : pd.Index(DATA_1D, dtype=float),
    "pandas[numpy]-series" : pd.Series(DATA_1D, dtype=float),
    "pandas[arrow]-index"  : pd.Index(DATA_1D, dtype="float[pyarrow]"),
    "pandas[arrow]-series" : pd.Series(DATA_1D, dtype="float[pyarrow]"),
}  # fmt: skip
r"""Example data for testing."""


@pytest.mark.parametrize("upper_included", [True, False])
@pytest.mark.parametrize("lower_included", [True, False])
@pytest.mark.parametrize("bounds", BOUNDS, ids=str)
@pytest.mark.parametrize("mode", ["clip", "mask"])
@pytest.mark.parametrize("data", TENSORS.values())
def test_boundary_encoder2[D: (pd.Series, pd.DataFrame, np.ndarray, torch.Tensor)](
    *,
    data: D,
    mode: BoundaryEncoder.ClippingMode,
    bounds: tuple[float | None, float | None],
    lower_included: bool,
    upper_included: bool,
) -> None:
    r"""Test the boundary encoder."""
    # create the encoder
    encoder = BoundaryEncoder(
        bounds[0],
        bounds[1],
        mode=mode,
        lower_included=lower_included,
        upper_included=upper_included,
    )
    # fit the encoder
    encoder.fit(data)

    # compute the encoded data
    encoded = encoder.encode(data)
    assert type(encoded) is type(data)
    assert encoded.shape == data.shape
    assert encoded.dtype == data.dtype

    lb, ub = encoder.lower_bound, encoder.upper_bound
    lb_given: bool = pd.notna(lb)
    ub_given: bool = pd.notna(ub)
    lower_mask: np.ndarray | D
    upper_mask: np.ndarray | D

    nan_data = np.isnan(data)
    nan_encoded = np.isnan(encoded)

    match lb_given, mode, lower_included:
        case False, _, _:
            lower_mask = np.zeros_like(data, dtype=bool)
        case True, "clip", _:
            lower_mask = data <= lb
        case True, "mask", True:
            lower_mask = data < lb
        case True, "mask", False:
            lower_mask = data <= lb
        case _:
            raise ValueError(f"Unexpected combination: {lb=} {mode=} {lower_included=}")

    match ub_given, mode, upper_included:
        case False, _, _:
            upper_mask = np.zeros_like(data, dtype=bool)
        case True, "clip", _:
            upper_mask = data >= ub
        case True, "mask", True:
            upper_mask = data > ub
        case True, "mask", False:
            upper_mask = data >= ub
        case _:
            raise ValueError(f"Unexpected combination: {ub=} {mode=} {upper_included=}")

    match mode:
        case "clip":
            assert (nan_data == nan_encoded).all()
            assert ((encoded == ub) == upper_mask).all()
            assert ((encoded == lb) == lower_mask).all()
        case "mask":
            assert (nan_encoded == (nan_data | lower_mask | upper_mask)).all()
        case _:
            raise ValueError(f"Unexpected mode: {mode=}")


@pytest.mark.parametrize("example", TENSORS)
def test_boundary_encoder(example: str) -> None:
    r"""Test the boundary encoder."""
    data = TENSORS[example]
    encoder = BoundaryEncoder(-1.0, +1.0, mode="clip")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert isinstance(encoded, type(data))
    assert type(encoded) is type(data)
    assert encoded.shape == data.shape
    assert encoded.dtype == data.dtype
    assert ((encoded >= -1) & (encoded <= 1)).all()
    assert ((encoded == -1) == (data <= -1)).all()
    assert ((encoded == +1) == (data >= +1)).all()

    match encoded:
        case np.ndarray() as array:
            assert array.dtype == data.dtype
        case torch.Tensor() as tensor:
            assert tensor.device == data.device  # type: ignore[attr-defined]
            assert tensor.dtype == data.dtype
        case pd.Index() as index:
            assert index.name == data.name  # type: ignore[attr-defined]
        case pd.Series() as series:
            assert series.name == data.name  # type: ignore[attr-defined]
            assert series.index.equals(data.index)  # type: ignore[attr-defined]
        case pd.DataFrame() as df:
            assert df.columns.equals(data.columns)  # type: ignore[attr-defined]
            assert df.index.equals(data.index)  # type: ignore[attr-defined]
        case _:
            raise TypeError(f"Unexpected type: {type(data)}")

    # test mode="mask" (fixed bounds)
    encoder = BoundaryEncoder(-1.0, +1.0, mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    original_invalid = (data < -1) | (data > +1)
    encoded_valid = (encoded >= -1) & (encoded <= 1)
    encoded_missing = np.isnan(encoded)
    assert (encoded_missing ^ encoded_valid).all()
    assert (encoded_missing == original_invalid).all()

    # test mode="mask" (learned bounds)
    encoder = BoundaryEncoder(mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    assert not (np.isnan(encoded)).any()
    assert (data == encoded).all()
    assert (data == decoded).all()

    # encode some data that violates bounds
    data2 = data * 2
    encoded2 = encoder.encode(data2)
    xmin, xmax = data.min(), data.max()
    mask = (data2 >= xmin) & (data2 <= xmax)
    assert (encoded2[mask] == data2[mask]).all()
    assert np.isnan(encoded2[~mask]).all()

    # test half-open interval + clip
    encoder = BoundaryEncoder(0.0, None, mode="clip")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert (encoded >= 0).all()
    assert ((encoded == 0) == (data <= 0)).all()

    # test half-open unbounded interval + mask
    encoder = BoundaryEncoder(0.0, None, mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    encoded_missing = np.isnan(encoded)
    assert (encoded_missing ^ (encoded >= 0)).all()
    assert (encoded_missing == (data < 0)).all()

    # test half-open bounded interval + mask
    encoder = BoundaryEncoder(0.0, 1.0, mode="mask", lower_included=False)
    encoder.fit(data)
    encoded = encoder.encode(data)
    encoded_missing = np.isnan(encoded)
    original_invalid = (data <= 0) | (data > 1)
    assert (encoded_missing ^ (encoded > 0)).all()
    assert (encoded_missing == original_invalid).all()
