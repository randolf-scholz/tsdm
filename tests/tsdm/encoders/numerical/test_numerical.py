r"""Test numerical encoders."""

import logging
import pickle
from tempfile import TemporaryFile

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pytest import mark, skip
from typing_extensions import Any, Callable, TypeVar, assert_type

from tsdm.encoders.numerical import (
    BoundaryEncoder,
    LinearScaler,
    MinMaxScaler,
    StandardScaler,
    get_broadcast,
    get_reduced_axes,
)

__logger__ = logging.getLogger(__name__)
T = TypeVar(
    "T",
    Callable[..., pd.Series],
    Callable[..., pd.DataFrame],
    Callable[..., np.ndarray],
    Callable[..., torch.Tensor],
)
D = TypeVar("D", pd.Series, pd.DataFrame, np.ndarray, torch.Tensor)
E = TypeVar("E", StandardScaler, MinMaxScaler)


@mark.parametrize("shape", ((5, 2, 3, 4), (7,)), ids=str)
@mark.parametrize("axis", ((1, -1), (-1,), -2, 0, None, ()), ids=str)
def test_get_broadcast(
    shape: tuple[int, ...], axis: None | int | tuple[int, ...]
) -> None:
    """Test the get_broadcast function."""
    if (isinstance(axis, tuple) and any(abs(a) > len(shape) - 1 for a in axis)) or (
        isinstance(axis, int) and abs(axis) > len(shape)
    ):
        skip(f"{shape=} {axis=}")

    arr: NDArray = np.asarray(np.random.randn(*shape))

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


def test_reduce_axes() -> None:
    """Test the get_reduced_axes function."""
    axis: tuple[int, ...] = (-2, -1)
    assert get_reduced_axes(..., axis) == axis
    assert get_reduced_axes(0, axis) == (-1,)
    assert get_reduced_axes([1], axis) == (-1,)
    assert get_reduced_axes([1, 2], axis) == axis
    assert get_reduced_axes(slice(None), axis) == axis
    assert get_reduced_axes((), axis) == axis
    assert get_reduced_axes((1,), axis) == (-1,)
    assert get_reduced_axes((slice(None), 1), (-2, -1)) == (-2,)

    axis = (-4, -3, -2, -1)
    assert get_reduced_axes((..., 1, slice(None)), axis) == (-4, -3, -1)
    assert get_reduced_axes((1, ..., 1), axis) == (-3, -2)
    assert get_reduced_axes((1, ...), axis) == (-3, -2, -1)


@mark.parametrize(
    "data",
    (
        np.array([-2.0, -1.1, -1.0, -0.9, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0]),
        torch.tensor([-2.0, -1.1, -1.0, -0.9, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0]),
        pd.Series([-2.0, -1.1, -1.0, -0.9, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0]),
    ),
    ids=("numpy", "torch", "pandas"),
)
def test_boundary_encoder(data: D) -> None:
    """Test the boundary encoder."""
    # test clip + numpy
    encoder = BoundaryEncoder(-1, +1, mode="clip")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert all((encoded >= -1) & (encoded <= 1))
    assert (encoded == -1).sum() == (data <= -1).sum()
    assert (encoded == +1).sum() == (data >= +1).sum()

    match data:
        case torch.Tensor() as tensor:
            assert isinstance(encoded, torch.Tensor) and encoded.shape == tensor.shape
        case np.ndarray() as array:
            assert isinstance(encoded, np.ndarray) and encoded.shape == array.shape
        case pd.Series() as series:
            assert (
                isinstance(encoded, pd.Series)
                and encoded.shape == series.shape
                and encoded.name == series.name
                and encoded.index.equals(series.index)
            )
        case _:
            raise TypeError(f"Unexpected type: {type(data)}")

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
    decoded = encoder.decode(encoded)
    assert not any(np.isnan(encoded))
    assert all(data == encoded)
    assert all(data == decoded)

    # encode some data that violates bounds
    data2 = data * 2
    encoded2 = encoder.encode(data2)
    xmin, xmax = data.min(), data.max()
    mask = (data2 >= xmin) & (data2 <= xmax)
    assert all(encoded2[mask] == data2[mask])  # pyright: ignore
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


@mark.parametrize("encoder_type", (StandardScaler, MinMaxScaler))
@mark.parametrize("tensor_type", (pd.Series, pd.DataFrame, np.array, torch.tensor))
def test_scaler(encoder_type: type[E], tensor_type: T) -> None:
    r"""Check whether the Standardizer encoder works as expected."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing.")

    if tensor_type != pd.DataFrame:
        LOGGER.info("Testing without batching.")
        data = np.random.rand(3)
        X = tensor_type(data)
        encoder = encoder_type()
        encoder.fit(X)
        encoded = encoder.encode(X)
        decoded = encoder.decode(encoded)
        assert np.allclose(X, decoded)
        assert encoder.params[0].shape == (), f"{encoder.params}"

    if tensor_type == pd.Series:
        return

    LOGGER.info("Testing with single batch-dim.")
    data = np.random.rand(3, 5)
    X = tensor_type(data)
    encoder = encoder_type(axis=-1)
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (5,), f"{encoder.params}"

    LOGGER.info("Testing slicing.")
    encoder = encoder[2]  # select the third encoder
    match encoded:
        case pd.DataFrame() as df:
            Y = df.loc[:, 2]
        case _ as arr:
            Y = arr[:, 2]
    match X:
        case pd.DataFrame() as df:
            X = df.loc[:, 2]
        case _ as arr:
            X = arr[:, 2]
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(Y, encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == ()

    if tensor_type == pd.DataFrame:
        return

    LOGGER.info("Testing with many batch-dim.")
    # weird input
    data = np.random.rand(2, 3, 4, 5)
    X = tensor_type(data)
    encoder = encoder_type(axis=(-2, -1))
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert encoder.params[0].shape == (4, 5), f"{encoder.params}"

    # encoder = encoder[:-1]  # select the first two component
    # # encoder.fit(X)
    # encoded = encoder.encode(X[:-1])
    # decoded = encoder.decode(encoded)
    # assert np.allclose(X, decoded)
    # assert encoder.params[0].shape == (2, 5)

    LOGGER.info("Testing finished!")


@mark.parametrize("encoder_type", (StandardScaler, MinMaxScaler))
def test_scaler_dataframe(encoder_type: type[E]) -> None:
    """Check whether the scaler-encoders work as expected on DataFrame."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing Encoder applied to pandas.DataFrame.")

    LOGGER.info("Testing without batching.")
    X = pd.DataFrame(np.random.rand(5, 3), columns=["a", "b", "c"])
    encoder = encoder_type(axis=-1)

    # validate fitting
    encoder.fit(X)
    assert encoder.params[0].shape == (3,), f"{encoder.params}"

    # validate encoding
    encoded = encoder.encode(X)
    assert (
        isinstance(encoded, pd.DataFrame)
        and encoded.shape == X.shape
        and encoded.columns.equals(X.columns)
        and encoded.index.equals(X.index)
    )
    if encoder_type is MinMaxScaler:
        assert all(encoded.min() >= 0.0)
        assert all(encoded.max() <= 1.0)
    if encoder_type is StandardScaler:
        assert np.allclose(encoded.mean(), 0.0)
        assert np.allclose(encoded.std(ddof=0), 1.0)

    # validate decoding
    decoded = encoder.decode(encoded)
    pd.testing.assert_frame_equal(X, decoded)
    assert np.allclose(X, decoded)

    # check parameters
    params = encoder.params
    assert isinstance(params, tuple)

    # test pickling
    with TemporaryFile() as f:
        pickle.dump(params, f)
        f.seek(0)
        reloaded_params = pickle.load(f)
        for x, y in zip(params, reloaded_params, strict=True):
            try:
                assert all(x == y)
            except Exception:
                assert x == y


@mark.parametrize("encoder_type", (StandardScaler, MinMaxScaler))
def test_scaler_series(encoder_type: type[E]) -> None:
    """Check whether the scaler-encoders work as expected on Series."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing Encoder applied to pandas.Series.")

    X = pd.Series([-1.0, 1.2, 2.7, 3.0], name="foo")
    encoder = encoder_type()

    # validate fitting
    encoder.fit(X)
    assert encoder.params[0].shape == (), f"{encoder.params}"

    # validate encoding
    encoded = encoder.encode(X)
    assert (
        isinstance(encoded, pd.Series)
        and encoded.shape == X.shape
        and encoded.name == X.name
        and encoded.index.equals(X.index)
    )
    if encoder_type is MinMaxScaler:
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0
    if encoder_type is StandardScaler:
        assert np.allclose(encoded.mean(), 0.0)
        assert np.allclose(encoded.std(ddof=0), 1.0)

    # validate decoding
    decoded = encoder.decode(encoded)
    pd.testing.assert_series_equal(X, decoded)
    assert np.allclose(X, decoded)


@mark.parametrize("axis", (None, (-2, -1), -1, ()), ids=lambda x: f"axis={x}")
def test_standard_scaler(axis):
    """Test the MinMaxScaler."""
    TRUE_SHAPE = {
        (): (),
        (-2, -1): (4, 5),
        -1: (5,),
        None: (2, 3, 4, 5),
    }[axis]

    # initialize
    X = np.random.randn(2, 3, 4, 5)
    encoder = StandardScaler(axis=axis)
    assert_type(encoder, StandardScaler[Any])

    # fit to data
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
def test_minmax_scaler(axis):
    """Test the MinMaxScaler."""
    TRUE_SHAPE = {
        None: (2, 3, 4, 5),
        (-2, -1): (4, 5),
        -1: (5,),
        (): (),
    }[axis]

    X = np.random.randn(2, 3, 4, 5)
    encoder = MinMaxScaler(axis=axis)
    assert_type(encoder, MinMaxScaler[Any])

    encoder.fit(X)
    assert encoder.params[0].shape == TRUE_SHAPE
    encoded = encoder.encode(X)
    assert encoded.min() >= 0.0
    assert encoded.max() <= 1.0
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)


@mark.parametrize("tensor_type", (pd.Series, pd.DataFrame, np.array, torch.tensor))
def test_linear_scaler(tensor_type: T) -> None:
    r"""Check whether the Standardizer encoder works as expected."""
    LOGGER = __logger__.getChild(LinearScaler.__name__)
    encoder_type = LinearScaler
    LOGGER.info("Testing.")

    LOGGER.info("Testing without batching.")
    data = np.random.rand(3)
    X = tensor_type(data)
    encoder = encoder_type()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert isinstance(encoder.params[0], float), f"{encoder.params}"

    if tensor_type == pd.Series:
        return

    LOGGER.info("Testing with single batch-dim.")
    data = np.random.rand(3, 5)
    X = tensor_type(data)
    encoder = encoder_type()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert isinstance(encoder.params[0], float), f"{encoder.params}"

    if tensor_type == pd.DataFrame:
        return

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
    data = np.random.rand(1, 2, 3, 4, 5)
    X = tensor_type(data)
    encoder = encoder_type()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert isinstance(encoder.params[0], float), f"{encoder.params}"

    # encoder = encoder[:-1]  # select the first two components
    # # encoder.fit(X)
    # encoded = encoder.encode(X[:-1])
    # decoded = encoder.decode(encoded)
    # assert np.allclose(X, decoded)
    # assert encoder.params[0].shape == (2, 3)
