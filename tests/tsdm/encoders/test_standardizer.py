#!/usr/bin/env python
r"""Test the standardizer encoder."""

import logging

import numpy as np
import torch
from pytest import mark

from tsdm.encoders import MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@mark.parametrize("Encoder", (StandardScaler, MinMaxScaler))
@mark.parametrize("tensor_type", (np.array, torch.tensor))
def test_standardizer(Encoder, tensor_type):
    r"""Check whether the Standardizer encoder works as expected."""
    LOGGER = __logger__.getChild(Encoder.__name__)
    LOGGER.info("Testing.")

    LOGGER.info("Testing without batching.")
    X = np.random.rand(3)
    X = tensor_type(X)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.param[0].shape == (), f"{encoder.param}"

    LOGGER.info("Testing with single batch-dim.")
    X = np.random.rand(3, 5)
    X = tensor_type(X)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.param[0].shape == (5,), f"{encoder.param}"

    LOGGER.info("Testing slicing.")
    encoder = encoder[2]  # select the third encoder
    # encoder.fit(X[:, 2])
    encoded = encoder.encode(X[:, 2])
    decoded = encoder.decode(encoded)
    assert np.allclose(X[:, 2], decoded)
    assert encoder.param[0].shape == ()

    LOGGER.info("Testing with many batch-dim.")
    # weird input
    X = np.random.rand(1, 2, 3, 4, 5)
    X = tensor_type(X)
    encoder = Encoder(axis=(1, 2, -1))
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.param[0].shape == (2, 3, 5), f"{encoder.param}"

    # encoder = encoder[:-1]  # select the first two components
    # # encoder.fit(X)
    # encoded = encoder.encode(X[:-1])
    # decoded = encoder.decode(encoded)
    # assert np.allclose(X, decoded)
    # assert encoder.param[0].shape == (2, 3)

    LOGGER.info("Testing finished!")


def _main() -> None:
    test_standardizer(StandardScaler, np.array)
    test_standardizer(MinMaxScaler, np.array)


if __name__ == "__main__":
    _main()
