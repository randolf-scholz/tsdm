r"""Tests for tsdm.encoders.box_cox."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tsdm.encoders.box_cox import BoxCoxEncoder, LogitBoxCoxEncoder

NAN = float("nan")

_NON_NEGATIVE_DATA = np.linspace(0, 20, 1000)
_NON_NEGATIVE_DATA_SPARSE = np.array([NAN, NAN, 1.0, 0.0, 10.0, 100.0, 0.0, NAN])
_LOGIT_DATA = np.linspace(0, 1, 20)
_LOGIT_DATA_SPARSE = np.array([NAN, NAN, 0.0, 1.0, 0.5, 0.0, 1.0, NAN])

NON_NEGATIVE_EXAMPLES = {
    "numpy"                 : _NON_NEGATIVE_DATA,
    "pandas[numpy]-index"   : pd.Index(_NON_NEGATIVE_DATA),
    "pandas[numpy]-series"  : pd.Series(_NON_NEGATIVE_DATA),
    "pandas[arrow]-index"   : pd.Index(_NON_NEGATIVE_DATA, dtype="float[pyarrow]"),
    "pandas[arrow]-series"  : pd.Series(_NON_NEGATIVE_DATA, dtype="float[pyarrow]"),
    "polars-series"         : pl.Series("a", _NON_NEGATIVE_DATA),
}  # fmt: skip

NON_NEGATIVE_SPARSE_EXAMPLES = {
    "numpy"                : _NON_NEGATIVE_DATA_SPARSE,
    "pandas[numpy]-index"  : pd.Index(_NON_NEGATIVE_DATA_SPARSE),
    "pandas[numpy]-series" : pd.Series(_NON_NEGATIVE_DATA_SPARSE),
    "pandas[arrow]-index"  : pd.Index(_NON_NEGATIVE_DATA_SPARSE, dtype="float[pyarrow]"),
    "pandas[arrow]-series" : pd.Series(_NON_NEGATIVE_DATA_SPARSE, dtype="float[pyarrow]"),
    "polars-series"        : pl.Series("a", _NON_NEGATIVE_DATA_SPARSE),
}  # fmt: skip


BOX_COX_EXAMPLES = {
    "dense": NON_NEGATIVE_EXAMPLES,
    "sparse": NON_NEGATIVE_SPARSE_EXAMPLES,
}

LOGIT_EXAMPLES = {
    "numpy-dense"          : _LOGIT_DATA,
    "numpy-sparse"         : _LOGIT_DATA_SPARSE,
    "pandas-dense-index"   : pd.Index(_LOGIT_DATA),
    "pandas-dense-series"  : pd.Series(_LOGIT_DATA),
    "pandas-sparse-index"  : pd.Index(_LOGIT_DATA_SPARSE),
    "pandas-sparse-series" : pd.Series(_LOGIT_DATA_SPARSE),
    "polars-dense-series"  : pl.Series("a", _LOGIT_DATA),
    "polars-sparse-series" : pl.Series("a", _LOGIT_DATA_SPARSE),
}  # fmt: skip


@pytest.mark.parametrize("method", BoxCoxEncoder.METHODS)
@pytest.mark.parametrize("example", NON_NEGATIVE_EXAMPLES)
@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_box_cox_encoder(
    example: str, method: BoxCoxEncoder.METHODS, kind: str
) -> None:
    r"""Test BoxCoxEncoder."""
    data = BOX_COX_EXAMPLES[kind][example]
    encoder = BoxCoxEncoder(method=method)

    encoder.fit(data)
    encoded = encoder.encode(data)
    assert isinstance(encoded, type(data))
    decoded = encoder.decode(encoded)
    assert isinstance(decoded, type(data))

    # ensure NANs are preserved
    assert all(np.isnan(data) == np.isnan(encoded))
    assert all(np.isnan(data) == np.isnan(decoded))

    # check that non-NAN values are close
    original = data[~np.isnan(data)]
    decoded = decoded[~np.isnan(decoded)]
    assert np.allclose(original, decoded), f"Max error:{max(abs(original - decoded))}"
    assert all(decoded >= 0.0)
    assert np.allclose(decoded.min(), 0.0)


@pytest.mark.parametrize("method", LogitBoxCoxEncoder.METHODS)
@pytest.mark.parametrize("example", LOGIT_EXAMPLES)
def test_logit_box_cox_encoder(
    example: str, method: LogitBoxCoxEncoder.METHODS
) -> None:
    r"""Test LogitBoxCoxEncoder."""
    data = LOGIT_EXAMPLES[example]
    encoder = LogitBoxCoxEncoder(method=method)

    encoder.fit(data)
    encoded = encoder.encode(data)
    assert isinstance(encoded, type(data))
    decoded = encoder.decode(encoded)
    assert isinstance(decoded, type(data))

    assert np.allclose(data, decoded), max(abs(data - decoded))
    assert all(decoded <= 1.0)
    assert np.allclose(decoded.max(), 1.0)
    assert all(decoded >= 0.0)
    assert np.allclose(decoded.min(), 0.0)
