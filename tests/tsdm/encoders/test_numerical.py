#!/usr/bin/env python
r"""Test the standardizer encoder.







"""

import logging

import numpy as np
import torch
from pytest import mark, skip

from tsdm.encoders.numerical import (
    BoundaryEncoder,
    LinearScaler,
    MinMaxScaler,
    StandardScaler,
    get_broadcast,
    get_reduced_axes,
)

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@mark.parametrize("shape", ((5, 2, 3, 4), (7,)))
@mark.parametrize("axis", ((1, -1), (-1,), -2, 0, None, ()))
def test_get_broadcast(
    shape: tuple[int, ...], axis: None | int | tuple[int, ...]
) -> None:
    """Test the get_broadcast function."""
    if (
        isinstance(axis, tuple) and max(map(abs, axis), default=0) > len(shape) - 1
    ) or (isinstance(axis, int) and abs(axis) > len(shape)):
        skip(f"{shape=} {axis=}")

    arr = np.random.randn(*shape)

    broadcast = get_broadcast(arr.shape, axis=axis)
    m = np.mean(arr, axis=axis)
    m_ref = np.mean(arr, axis=axis, keepdims=True)
    assert m[broadcast].shape == m_ref.shape

    # test with keep_axis:
    kept_axis = axis
    broadcast = get_broadcast(arr.shape, axis=kept_axis, keep_axis=True)
    match kept_axis:
        case None:
            contracted_axes = tuple(range(arr.ndim))
        case int():
            contracted_axes = tuple(set(range(arr.ndim)) - {kept_axis % arr.ndim})
        case tuple():
            contracted_axes = tuple(
                set(range(arr.ndim)) - {a % arr.ndim for a in kept_axis}
            )
    m = np.mean(arr, axis=contracted_axes)
    m_ref = np.mean(arr, axis=contracted_axes, keepdims=True)
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


def test_boundary_encoder() -> None:
    """Test the boundary encoder."""
    data = np.array([-2.0, -1.1, -1.0, -0.9, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0])

    # test clip + numpy
    encoder = BoundaryEncoder(-1, +1, mode="clip")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert all((encoded >= -1) & (encoded <= 1))
    assert (encoded == -1).sum() == (data <= -1).sum()
    assert (encoded == +1).sum() == (data >= +1).sum()

    # test numpy + mask
    encoder = BoundaryEncoder(-1, +1, mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert all(np.isnan(encoded) ^ ((encoded >= -1) & (encoded <= 1)))
    assert np.isnan(encoded).sum() == ((data < -1).sum() + (data > +1).sum())

    # test fitting with mask
    encoder = BoundaryEncoder(mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert not any(np.isnan(encoded))
    assert all(data == encoded)

    # encode some data that violates bounds
    data2 = data * 2
    encoded2 = encoder.encode(data2)
    xmin, xmax = data.min(), data.max()
    mask = (data2 >= xmin) & (data2 <= xmax)
    assert all(encoded2[mask] == data2[mask])
    assert all(np.isnan(encoded2[~mask]))

    # test half-open interval + clip
    encoder = BoundaryEncoder(0, None, mode="clip")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert all(encoded >= 0)
    assert (encoded == 0).sum() == (data <= 0).sum()

    # test half-open unbounded interval + mask
    encoder = BoundaryEncoder(0, None, mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert all(np.isnan(encoded) ^ (encoded >= 0))
    assert np.isnan(encoded).sum() == (data < 0).sum()

    # test half-open bounded interval + mask
    encoder = BoundaryEncoder(0, 1, mode="mask", lower_included=False)
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert all(np.isnan(encoded) ^ (encoded > 0))
    assert np.isnan(encoded).sum() == ((data <= 0).sum() + (data > 1).sum())


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
    encoder = Encoder(axis=-1)
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


@mark.parametrize("axis", (None, (-2, -1), -1, ()), ids=lambda x: f"axis={x}")
def test_standard_scaler(axis) -> None:
    """Test the MinMaxScaler."""
    TRUE_SHAPE = {
        None: (2, 3, 4, 5),
        (-2, -1): (4, 5),
        -1: (5,),
        (): (),
    }[axis]

    X = np.random.randn(2, 3, 4, 5)
    encoder = StandardScaler(axis=axis)
    encoder.fit(X)
    assert encoder.params[0].shape == TRUE_SHAPE
    encoded = encoder.encode(X)

    if axis is None:
        # std = 0.0
        return

    assert np.allclose(encoded.mean(), 0.0)
    assert np.allclose(encoded.std(), 1.0)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)


@mark.parametrize("axis", (None, (-2, -1), -1, ()), ids=lambda x: f"axis={x}")
def test_minmax_scaler(axis) -> None:
    """Test the MinMaxScaler."""
    TRUE_SHAPE = {
        None: (2, 3, 4, 5),
        (-2, -1): (4, 5),
        -1: (5,),
        (): (),
    }[axis]

    X = np.random.randn(2, 3, 4, 5)
    encoder = MinMaxScaler(axis=axis)
    encoder.fit(X)
    assert encoder.params[0].shape == TRUE_SHAPE
    encoded = encoder.encode(X)
    assert encoded.min() >= 0.0
    assert encoded.max() <= 1.0
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)


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
