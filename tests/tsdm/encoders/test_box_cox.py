r"""Tests for tsdm.encoders.box_cox."""

import numpy as np
import pandas as pd
import pytest

from tsdm.encoders.box_cox import BoxCoxEncoder, LogitBoxCoxEncoder

NAN = float("nan")

_NON_NEGATIVE_DATA = np.linspace(0, 20, 1000)
_NON_NEGATIVE_DATA_SPARSE = np.array([NAN, NAN, 1.0, 0.0, 10.0, 100.0, 0.0, NAN])
_LOGIT_DATA = np.linspace(0, 1, 20)
_LOGIT_DATA_SPARSE = np.array([NAN, NAN, 0.0, 1.0, 0.5, 0.0, 0.3, NAN])

BOX_BOX_EXAMPLES_DENSE = {
    "numpy"                 : _NON_NEGATIVE_DATA,
    "pandas[numpy]-index"   : pd.Index(_NON_NEGATIVE_DATA),
    "pandas[numpy]-series"  : pd.Series(_NON_NEGATIVE_DATA),
    "pandas[arrow]-index"   : pd.Index(_NON_NEGATIVE_DATA, dtype="float[pyarrow]"),
    "pandas[arrow]-series"  : pd.Series(_NON_NEGATIVE_DATA, dtype="float[pyarrow]"),
}  # fmt: skip

BOX_COX_EXAMPLES_SPARSE = {
    "numpy"                : _NON_NEGATIVE_DATA_SPARSE,
    "pandas[numpy]-index"  : pd.Index(_NON_NEGATIVE_DATA_SPARSE),
    "pandas[numpy]-series" : pd.Series(_NON_NEGATIVE_DATA_SPARSE),
    "pandas[arrow]-index"  : pd.Index(_NON_NEGATIVE_DATA_SPARSE, dtype="float[pyarrow]"),
    "pandas[arrow]-series" : pd.Series(_NON_NEGATIVE_DATA_SPARSE, dtype="float[pyarrow]"),
}  # fmt: skip


BOX_COX_EXAMPLES = {
    "dense"  : BOX_BOX_EXAMPLES_DENSE,
    "sparse" : BOX_COX_EXAMPLES_SPARSE,
}  # fmt: skip

LOGIT_EXAMPLES_DENSE = {
    "numpy"                : _LOGIT_DATA,
    "pandas[numpy]-index"  : pd.Index(_LOGIT_DATA),
    "pandas[numpy]-series" : pd.Series(_LOGIT_DATA),
    "pandas[arrow]-index"  : pd.Index(_LOGIT_DATA, dtype="float[pyarrow]"),
    "pandas[arrow]-series" : pd.Series(_LOGIT_DATA, dtype="float[pyarrow]"),
}  # fmt: skip

LOGIT_EXAMPLES_SPARSE = {
    "numpy"                : _LOGIT_DATA_SPARSE,
    "pandas[arrow]-index"  : pd.Index(_LOGIT_DATA_SPARSE, dtype="float[pyarrow]"),
    "pandas[arrow]-series" : pd.Series(_LOGIT_DATA_SPARSE, dtype="float[pyarrow]"),
    "pandas[numpy]-index"  : pd.Index(_LOGIT_DATA_SPARSE),
    "pandas[numpy]-series" : pd.Series(_LOGIT_DATA_SPARSE),
}  # fmt: skip


LOGIT_EXAMPLES = {
    "dense"  : LOGIT_EXAMPLES_DENSE,
    "sparse" : LOGIT_EXAMPLES_SPARSE,
}  # fmt: skip


@pytest.mark.parametrize("method", BoxCoxEncoder.METHOD)
@pytest.mark.parametrize("example", BOX_BOX_EXAMPLES_DENSE)
@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_box_cox_encoder(example: str, kind: str, method: BoxCoxEncoder.METHOD) -> None:
    r"""Test BoxCoxEncoder."""
    data = BOX_COX_EXAMPLES[kind][example]
    encoder = BoxCoxEncoder(method=method)

    # check that encoding raises error before fitting
    if method is not BoxCoxEncoder.METHOD.fixed:
        with pytest.raises(RuntimeError):
            encoder.encode(data)
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


@pytest.mark.parametrize("method", LogitBoxCoxEncoder.METHOD)
@pytest.mark.parametrize("example", LOGIT_EXAMPLES_DENSE)
@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_logit_box_cox_encoder(
    example: str, kind: str, method: LogitBoxCoxEncoder.METHOD
) -> None:
    r"""Test LogitBoxCoxEncoder."""
    data = LOGIT_EXAMPLES[kind][example]
    encoder = LogitBoxCoxEncoder(method=method)

    # check that encoding raises error before fitting
    if method is not LogitBoxCoxEncoder.METHOD.fixed:
        with pytest.raises(RuntimeError):
            encoder.encode(data)
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
    assert np.allclose(original, decoded), max(abs(original - decoded))
    assert all(decoded <= 1.0)
    assert np.allclose(decoded.max(), 1.0)
    assert all(decoded >= 0.0)
    assert np.allclose(decoded.min(), 0.0)
