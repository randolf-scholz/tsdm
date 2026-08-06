r"""Tests for Boundary Encoders."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from tsdm.encoders import BoundaryEncoder

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
BOUNDS: list[tuple[float | None, float | None]] = [
    (-1, +1),
    (0, 1),
    (0, float("inf")),
    (0, None),
    (0, float("nan")),
]

TENSORS: dict[str, Any] = {
    "numpy-1D"             : np.array(DATA_1D),
    "torch-1D"             : torch.tensor(DATA_1D),
    # "pandas[numpy]-index"  : pd.Index(DATA_1D, dtype=float),
    "pandas[numpy]-series" : pd.Series(DATA_1D, dtype=float),
    # "pandas[arrow]-index"  : pd.Index(DATA_1D, dtype="float[pyarrow]"),
    "pandas[arrow]-series" : pd.Series(DATA_1D, dtype="float[pyarrow]"),
}  # fmt: skip
r"""Example data for testing."""


@pytest.mark.parametrize(
    ("valid", "invalid"),
    [
        (np.array(DATA_1D), np.array([[0.0, 1.0], [2.0, 3.0]])),
        (pd.Series(DATA_1D), pd.DataFrame([[0.0, 1.0], [2.0, 3.0]])),
        (torch.tensor(DATA_1D), torch.tensor([[0.0, 1.0], [2.0, 3.0]])),
    ],
    ids=["numpy", "pandas", "torch"],
)
def test_boundary_encoder_rejects_non_1d(valid: Any, invalid: Any) -> None:
    r"""Test that BoundaryEncoder rejects non-one-dimensional data."""
    encoder = BoundaryEncoder()

    with pytest.raises(ValueError, match="expected one-dimensional data"):
        encoder.fit(invalid)

    encoder.fit(valid)


@pytest.mark.parametrize("upper_included", [True, False])
@pytest.mark.parametrize("lower_included", [True, False])
@pytest.mark.parametrize("bounds", BOUNDS, ids=str)
@pytest.mark.parametrize("mode", ["clip", "mask"])
# @pytest.mark.parametrize("data", TENSORS.values())
@pytest.mark.parametrize("case", TENSORS)
def test_boundary_encoder2(
    case: str,
    *,
    mode: str,
    bounds: tuple[float | None, float | None],
    lower_included: bool,
    upper_included: bool,
) -> None:
    r"""Test the boundary encoder."""
    data = TENSORS[case]
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
    nan_data = np.isnan(data)
    nan_encoded = np.isnan(encoded)

    match mode, lower_included:
        case _ if lb is None or pd.isna(lb):
            lower_mask = np.zeros_like(data, dtype=bool)
        case "clip", _:
            lower_mask = data <= lb
        case "mask", True:
            lower_mask = data < lb
        case "mask", False:
            lower_mask = data <= lb
        case _:
            raise ValueError(f"Unexpected combination: {mode=} {lower_included=}")

    match mode, upper_included:
        case _ if ub is None or pd.isna(ub):
            upper_mask = np.zeros_like(data, dtype=bool)
        case "clip", _:
            upper_mask = data >= ub
        case "mask", True:
            upper_mask = data > ub
        case "mask", False:
            upper_mask = data >= ub
        case _:
            raise ValueError(f"Unexpected combination: {mode=} {upper_included=}")

    match mode:
        case "clip":
            assert (nan_data == nan_encoded).all()
            assert ((encoded == ub) == upper_mask).all()
            assert ((encoded == lb) == lower_mask).all()
        case "mask":
            assert (nan_encoded == (nan_data | lower_mask | upper_mask)).all()
        case _:
            raise ValueError(f"Unexpected mode: {mode=}")


@pytest.mark.parametrize("case", TENSORS)
def test_boundary_encoder(case: str) -> None:
    r"""Test the boundary encoder."""
    data = TENSORS[case]
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

    # match encoded, data:
    #     case np.ndarray() as transformed, np.ndarray() as original:
    #         assert transformed.dtype == original.dtype
    #     case torch.Tensor() as transformed, torch.Tensor() as original:
    #         assert transformed.device == original.device
    #         assert transformed.dtype == data.dtype
    #     case pd.Index() as transformed, pd.Index() as original:
    #         assert transformed.name == original.name
    #     case pd.Series() as transformed, pd.Series() as original:
    #         assert transformed.name == original.name
    #         assert transformed.index.equals(original.index)
    #     case pd.DataFrame() as transformed, pd.DataFrame() as original:
    #         assert transformed.columns.equals(original.columns)
    #         assert transformed.index.equals(original.index)
    #     case _:
    #         raise TypeError(f"Unexpected type: {type(data)}")

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

    # test half-open interval with clip
    encoder = BoundaryEncoder(0.0, None, mode="clip")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert (encoded >= 0).all()
    assert ((encoded == 0) == (data <= 0)).all()

    # test half-open unbounded interval with mask
    encoder = BoundaryEncoder(0.0, None, mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    encoded_missing = np.isnan(encoded)
    assert (encoded_missing ^ (encoded >= 0)).all()
    assert (encoded_missing == (data < 0)).all()

    # test half-open and bounded interval with mask
    encoder = BoundaryEncoder(0.0, 1.0, mode="mask", lower_included=False)
    encoder.fit(data)
    encoded = encoder.encode(data)
    encoded_missing = np.isnan(encoded)
    original_invalid = (data <= 0) | (data > 1)
    assert (encoded_missing ^ (encoded > 0)).all()
    assert (encoded_missing == original_invalid).all()
