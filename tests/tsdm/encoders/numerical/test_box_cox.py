"""Tests for tsdm.encoders.box_cox."""

import numpy as np
from pytest import mark

from tsdm.encoders.box_cox import METHODS, BoxCoxEncoder, LogitBoxCoxEncoder


@mark.parametrize("method", METHODS)
def test_box_cox_encoder(method):
    """Test BoxCoxEncoder."""
    data = np.linspace(0, 1000, 1000)
    encoder = BoxCoxEncoder(method=method)

    encoder.fit(data)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)

    assert np.allclose(data, decoded), f"Max error:{max(abs(data - decoded))}"
    assert all(decoded >= 0.0)
    assert np.allclose(decoded.min(), 0.0)


@mark.parametrize("method", METHODS)
def test_logit_box_cox_encoder(method):
    """Test LogitBoxCoxEncoder."""
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
    # not sure why doesn't quite reach 0.0
