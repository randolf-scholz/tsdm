r"""#TODO add module summary line.

#TODO add module description.
"""

import logging

import numpy as np
from pytest import mark

from tsdm.encoders.modular import MinMaxScaler, Standardizer

__logger__ = logging.getLogger(__name__)


@mark.parametrize("Encoder", (Standardizer, MinMaxScaler))
def test_standardizer(Encoder):
    """Check whether the Standardizer encoder works as expected."""
    X = np.random.rand(3, 5)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.param[0].shape == (5,)

    X = np.random.rand(3)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.param[0].shape == ()
