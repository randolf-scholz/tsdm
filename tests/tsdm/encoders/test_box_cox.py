r"""Tests for tsdm.encoders.box_cox."""

import numpy as np
import pytest

from tsdm.encoders.box_cox import BoxCoxEncoder, LogitBoxCoxEncoder


@pytest.mark.parametrize("method", BoxCoxEncoder.METHODS)
def test_box_cox_encoder(method: BoxCoxEncoder.METHODS) -> None:
    r"""Test BoxCoxEncoder."""
    data = np.linspace(0, 1000, 1000)
    encoder = BoxCoxEncoder(method=method)

    encoder.fit(data)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)

    assert np.allclose(data, decoded), f"Max error:{max(abs(data - decoded))}"
    assert all(decoded >= 0.0)
    assert np.allclose(decoded.min(), 0.0)


@pytest.mark.parametrize("method", LogitBoxCoxEncoder.METHODS)
def test_logit_box_cox_encoder(method: LogitBoxCoxEncoder.METHODS) -> None:
    r"""Test LogitBoxCoxEncoder."""
    data = np.linspace(0, 1, 1000)
    encoder = LogitBoxCoxEncoder(method=method)

    encoder.fit(data)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)

    assert np.allclose(data, decoded), max(abs(data - decoded))
    assert all(decoded <= 1.0)
    assert np.allclose(decoded.max(), 1.0)
    assert all(decoded >= 0.0)
    assert np.allclose(decoded.min(), 0.0)
