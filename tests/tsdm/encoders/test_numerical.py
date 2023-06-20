#!/usr/bin/env python
r"""Test the standardizer encoder."""

import logging

import numpy as np
import torch
from pytest import mark

from tsdm.encoders.numerical import (
    LinearScaler,
    MinMaxScaler,
    StandardScaler,
    get_broadcast,
    get_reduced_axes,
)

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@mark.parametrize("shape", ((1, 2, 3, 4), (7,)))
@mark.parametrize("axis", ((1, -1), (-1,), 0, None, ()))
def test_get_broadcast(
    shape: tuple[int, ...], axis: None | int | tuple[int, ...]
) -> None:
    """Test the get_broadcast function."""
    if isinstance(axis, tuple) and len(axis) > len(shape):
        return

    arr = np.random.randn(*shape)
    broadcast = get_broadcast(arr.shape, axis=axis)
    m = np.mean(arr, axis=axis)
    m_ref = np.mean(arr, axis=axis, keepdims=True)
    assert m[broadcast].shape == m_ref.shape


def test_reduce_axes():
    """Test the get_reduced_axes function."""
    axes: tuple[int, ...] = (-2, -1)
    assert get_reduced_axes(..., axes) == axes
    assert get_reduced_axes(0, axes) == (-1,)
    assert get_reduced_axes([1], axes) == (-1,)
    assert get_reduced_axes([1, 2], axes) == axes
    assert get_reduced_axes(slice(None), axes) == axes
    assert get_reduced_axes((), axes) == axes
    assert get_reduced_axes((1,), axes) == (-1,)
    assert get_reduced_axes((slice(None), 1), (-2, -1)) == (-2,)

    axes = (-4, -3, -2, -1)
    assert get_reduced_axes((..., 1, slice(None)), axes) == (-4, -3, -1)
    assert get_reduced_axes((1, ..., 1), axes) == (-3, -2)
    assert get_reduced_axes((1, ...), axes) == (-3, -2, -1)


@mark.parametrize("Encoder", (StandardScaler, MinMaxScaler))
@mark.parametrize("tensor_type", (np.array, torch.tensor))
def test_scaler(Encoder, tensor_type):
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
    assert encoder.params[0].shape == (), f"{encoder.params}"

    LOGGER.info("Testing with single batch-dim.")
    X = np.random.rand(3, 5)
    X = tensor_type(X)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (5,), f"{encoder.params}"

    LOGGER.info("Testing slicing.")
    encoder = encoder[2]  # select the third encoder
    Y = encoded
    # encoder.fit(X[:, 2])
    encoded = encoder.encode(X[:, 2])
    decoded = encoder.decode(encoded)
    assert np.allclose(Y[:, 2], encoded)
    assert np.allclose(X[:, 2], decoded)
    assert encoder.params[0].shape == ()

    LOGGER.info("Testing with many batch-dim.")
    # weird input
    X = np.random.rand(1, 2, 3, 4, 5)
    X = tensor_type(X)
    encoder = Encoder(axis=(1, 2, -1))
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (2, 3, 5), f"{encoder.params}"

    encoder = encoder[:-1]  # select the first two components
    # encoder.fit(X)
    encoded = encoder.encode(X[:-1])
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (1, 3, 5)

    LOGGER.info("Testing finished!")


def test_standard_scaler() -> None:
    """Test the StandardScaler."""
    X = np.random.rand(100)
    encoder = StandardScaler()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert np.allclose(encoded.mean(), 0.0)
    assert np.allclose(encoded.std(), 1.0)


def test_minmax_scaler() -> None:
    """Test the MinMaxScaler."""
    X = np.random.randn(100)
    encoder = MinMaxScaler()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoded.min() >= 0.0
    assert encoded.max() <= 1.0


@mark.parametrize("tensor_type", (np.array, torch.tensor))
def test_linear_scaler(tensor_type):
    r"""Check whether the Standardizer encoder works as expected."""
    LOGGER = __logger__.getChild(LinearScaler.__name__)
    Encoder = LinearScaler
    LOGGER.info("Testing.")

    LOGGER.info("Testing without batching.")
    X = np.random.rand(3)
    X = tensor_type(X)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (), f"{encoder.params}"

    LOGGER.info("Testing with single batch-dim.")
    X = np.random.rand(3, 5)
    X = tensor_type(X)
    encoder = Encoder()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (), f"{encoder.params}"

    # LOGGER.info("Testing slicing.")
    # encoder = encoder[2]  # select the third encoder
    # Y = encoded
    # # encoder.fit(X[:, 2])
    # encoded = encoder.encode(X[:, 2])
    # decoded = encoder.decode(encoded)
    # assert np.allclose(Y[:, 2], encoded)
    # assert np.allclose(X[:, 2], decoded)
    # assert encoder.params[0].shape == ()

    LOGGER.info("Testing with many batch-dim.")
    # weird input
    X = np.random.rand(1, 2, 3, 4, 5)
    X = tensor_type(X)
    encoder = Encoder(axis=(1, 2))
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (), f"{encoder.params}"

    # encoder = encoder[:-1]  # select the first two components
    # # encoder.fit(X)
    # encoded = encoder.encode(X[:-1])
    # decoded = encoder.decode(encoded)
    # assert np.allclose(X, decoded)
    # assert encoder.params[0].shape == (2, 3)


def _main() -> None:
    test_scaler(StandardScaler, np.array)
    test_scaler(MinMaxScaler, np.array)
    test_linear_scaler(np.array)
    test_standard_scaler()
    test_minmax_scaler()


if __name__ == "__main__":
    _main()
