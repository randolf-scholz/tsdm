#!/usr/bin/env python
r"""Test the standardizer encoder."""

import logging

import numpy as np
import torch
from pytest import mark

from tsdm.encoders import MinMaxScaler, Standardizer

__logger__ = logging.getLogger(__name__)


@mark.parametrize("Encoder", (Standardizer, MinMaxScaler))
@mark.parametrize("tensor_type", (np.array, torch.tensor))
def test_standardizer(Encoder, tensor_type):
    r"""Check whether the Standardizer encoder works as expected."""
    __logger__.info("Testing %s started!", Encoder.__name__)

    X = np.random.rand(3)
    X = tensor_type(X)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.param[0].shape == (), f"{encoder.param}"

    X = np.random.rand(3, 5)
    X = tensor_type(X)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.param[0].shape == (5,), f"{encoder.param}"

    encoder = encoder[2]  # select the third encoder
    # encoder.fit(X[:, 2])
    encoded = encoder.encode(X[:, 2])
    decoded = encoder.decode(encoded)
    assert np.allclose(X[:, 2], decoded)
    assert encoder.param[0].shape == ()

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

    __logger__.info("Testing %s finished!", Encoder.__name__)


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_standardizer(Standardizer, np.array)
    test_standardizer(MinMaxScaler, np.array)


if __name__ == "__main__":
    _main()
