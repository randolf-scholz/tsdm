r"""Test numerical encoders."""

import logging
import pickle
from collections.abc import Callable
from tempfile import TemporaryFile
from typing import Any, assert_type

import numpy as np
import pandas as pd
import pytest
import torch

from tsdm.constants import RNG
from tsdm.encoders.numerical import (
    LinearScaler,
    MinMaxScaler,
    StandardScaler,
    get_broadcast,
    reduce_axes,
)
from tsdm.types.aliases import Axis

__logger__ = logging.getLogger(__name__)

DATA_1D = [
    float("-inf"),
    -1.1,
    -1.0,
    -0.9,
    -0.0,
    +0.0,
    +0.3,
    +1.0,
    +1.5,
    float("+inf"),
]
DATA_2D = [
    [-2.0, -1.1, -1.0, -0.9],
    [ 0.0,  0.3,  0.5,  1.0],
    [ 1.5,  2.0,  2.5,  3.0],
]  # fmt: skip


@pytest.mark.parametrize(
    "tensor_type", [pd.Series, pd.DataFrame, np.array, torch.tensor]
)
def test_linear_scaler[
    T: (
        Callable[..., pd.Series],
        Callable[..., pd.DataFrame],
        Callable[..., np.ndarray],
        Callable[..., torch.Tensor],
    )
](tensor_type: T) -> None:
    r"""Check whether the Standardizer encoder works as expected."""
    LOGGER = __logger__.getChild(LinearScaler.__name__)
    encoder_type = LinearScaler
    LOGGER.info("Testing.")

    LOGGER.info("Testing without batching.")
    data = RNG.uniform(size=3)
    X = tensor_type(data)
    encoder = encoder_type()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert isinstance(encoder.params["loc"], float), f"{encoder.params}"

    if tensor_type == pd.Series:
        return

    LOGGER.info("Testing with single batch-dim.")
    data = RNG.uniform(size=(3, 5))
    X = tensor_type(data)
    encoder = encoder_type()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert isinstance(encoder.params["loc"], float), f"{encoder.params}"

    if tensor_type == pd.DataFrame:
        return

    LOGGER.info("Testing with many batch-dim.")
    data = RNG.uniform(size=(1, 2, 3, 4, 5))
    X = tensor_type(data)
    encoder = encoder_type()
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    assert isinstance(encoder.params["loc"], float), f"{encoder.params}"


@pytest.mark.parametrize("shape", [(5, 2, 3, 4), (7,)], ids=lambda x: f"shape={x}")
@pytest.mark.parametrize(
    "axis",
    [(1, -1), (-2,), (-1,), (0,), -2, -1, 0, None, ()],
    ids=lambda x: f"axis={x}",
)
def test_get_broadcast(shape: tuple[int, ...], axis: Axis) -> None:
    r"""Test the get_broadcast function."""
    # initialize array
    arr: np.ndarray = RNG.normal(size=shape)

    if (isinstance(axis, int) and abs(axis) > arr.ndim) or (
        isinstance(axis, tuple)
        and (len(axis) > arr.ndim or any(abs(a) > arr.ndim for a in axis))
    ):
        pytest.skip(f"Invalid shape axis combination: {shape=} {axis=}")

    broadcast = get_broadcast(arr.shape, axis=axis)
    m: np.ndarray = np.mean(arr, axis=axis)
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


@pytest.mark.parametrize(
    ("axis", "selection", "expected"),
    [
        ((-2, -1)        , ...                   , (-2, -1)     ),
        ((-2, -1)        , 0                     , (-1,)        ),
        ((-2, -1)        , [1]                   , (-1,)        ),
        ((-2, -1)        , [1, 2]                , (-2, -1)     ),
        ((-2, -1)        , slice(None)           , (-2, -1)     ),
        ((-2, -1)        , ()                    , (-2, -1)     ),
        ((-2, -1)        , (1,)                  , (-1,)        ),
        ((-2, -1)        , (slice(None), 1)      , (-2,)        ),
        ((-4, -3, -2, -1), (..., 1, slice(None)) , (-4, -3, -1) ),
        ((-4, -3, -2, -1), (1, ..., 1)           , (-3, -2)     ),
        ((-4, -3, -2, -1), (1, ...)              , (-3, -2, -1) ),
    ],
    ids=str,
)  # fmt: skip
def test_reduce_axes(
    axis: tuple[int, ...], selection: Any, expected: tuple[int, ...]
) -> None:
    r"""Test the `reduced_axes` function."""
    assert reduce_axes(axis, selection) == expected


@pytest.mark.parametrize("encoder_type", [StandardScaler, MinMaxScaler])
@pytest.mark.parametrize(
    "tensor_type", [pd.Series, pd.DataFrame, np.array, torch.tensor]
)
def test_scaler[
    T: (
        Callable[..., pd.Series],
        Callable[..., pd.DataFrame],
        Callable[..., np.ndarray],
        Callable[..., torch.Tensor],
    ),
    E: (StandardScaler, MinMaxScaler),
](encoder_type: type[E], tensor_type: T) -> None:
    r"""Check whether the Standardizer encoder works as expected."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing.")

    if tensor_type != pd.DataFrame:
        LOGGER.info("Testing without batching.")
        data = RNG.uniform(size=3)
        X = tensor_type(data)
        encoder = encoder_type()
        encoder.fit(X)
        encoded = encoder.encode(X)
        decoded = encoder.decode(encoded)
        assert np.allclose(X, decoded)
        # assert encoder.params["loc"].shape == (), f"{encoder.params}"

    if tensor_type == pd.Series:
        return

    LOGGER.info("Testing with single batch-dim.")
    data = RNG.uniform(size=(3, 5))
    X = tensor_type(data)
    encoder = encoder_type(axis=-1)
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    # assert encoder.params["loc"].shape == (5,), f"{encoder.params}"

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
    # assert encoder.params["loc"].shape == ()

    if tensor_type == pd.DataFrame:
        return

    LOGGER.info("Testing with many batch-dim.")
    data = RNG.uniform(size=(2, 3, 4, 5))
    X = tensor_type(data)
    encoder = encoder_type(axis=(-2, -1))
    encoder.fit(X)
    encoded = encoder.encode(X)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
    # assert encoder.params["loc"].shape == (4, 5), f"{encoder.params}"

    LOGGER.info("Testing finished!")


@pytest.mark.parametrize("encoder_type", [StandardScaler, MinMaxScaler])
def test_scaler_dataframe[
    E: (StandardScaler, MinMaxScaler),
](encoder_type: type[E]) -> None:
    r"""Check whether the scaler-encoders work as expected on DataFrame."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing Encoder applied to pandas.DataFrame.")

    LOGGER.info("Testing without batching.")
    X = pd.DataFrame(RNG.uniform(size=(5, 3)), columns=["a", "b", "c"])
    encoder = encoder_type(axis=-1)

    # validate fitting
    encoder.fit(X)

    # validate encoding
    encoded = encoder.encode(X)
    assert isinstance(encoded, pd.DataFrame)
    assert encoded.shape == X.shape
    assert encoded.columns.equals(X.columns)
    assert encoded.index.equals(X.index)

    if encoder_type is MinMaxScaler:
        assert encoder.params["ymin"].shape == (3,)
        assert all(encoded.min() >= 0.0)
        assert all(encoded.max() <= 1.0)
    if encoder_type is StandardScaler:
        assert encoder.params["mean"].shape == (3,)
        assert np.allclose(encoded.mean(), 0.0)
        assert np.allclose(encoded.std(ddof=0), 1.0)

    # validate decoding
    decoded = encoder.decode(encoded)
    pd.testing.assert_frame_equal(X, decoded)
    assert np.allclose(X, decoded)

    # check parameters
    params = encoder.params
    assert isinstance(params, dict)

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


@pytest.mark.parametrize("encoder_type", [StandardScaler, MinMaxScaler])
def test_scaler_series[
    E: (MinMaxScaler, StandardScaler),
](
    encoder_type: type[E],
) -> None:
    r"""Check whether the scaler-encoders work as expected on Series."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing Encoder applied to pandas.Series.")

    X = pd.Series([-1.0, 1.2, 2.7, 3.0], name="foo")
    encoder = encoder_type()

    # validate fitting
    encoder.fit(X)

    # validate encoding
    encoded = encoder.encode(X)
    assert isinstance(encoded, pd.Series)
    assert encoded.shape == X.shape
    assert encoded.name == X.name
    assert encoded.index.equals(X.index)

    # validate encoded values
    if encoder_type is MinMaxScaler:
        assert encoder.params["ymin"].shape == ()
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0
    if encoder_type is StandardScaler:
        assert encoder.params["mean"].shape == ()
        assert np.allclose(encoded.mean(), 0.0)
        assert np.allclose(encoded.std(ddof=0), 1.0)

    # validate decoding
    decoded = encoder.decode(encoded)
    pd.testing.assert_series_equal(X, decoded)
    assert np.allclose(X, decoded)


@pytest.mark.parametrize("axis", [None, (-2, -1), -1, ()], ids=lambda x: f"axis={x}")
def test_standard_scaler(axis: Axis) -> None:
    r"""Test the MinMaxScaler."""
    TRUE_SHAPES: dict[Axis, tuple[int, ...]] = {
        (): (),
        (-2, -1): (4, 5),
        -1: (5,),
        None: (2, 3, 4, 5),
    }
    expected = TRUE_SHAPES[axis]

    # initialize
    X = RNG.normal(size=(2, 3, 4, 5))
    encoder = StandardScaler(axis=axis)
    assert_type(encoder, StandardScaler[Any])

    # fit to data
    encoder.fit(X)
    result = encoder.params["mean"].shape
    assert result == expected
    encoded = encoder.encode(X)

    if axis is None:  # std = 0.0
        return

    assert np.allclose(encoded.mean(), 0.0)
    assert np.allclose(encoded.std(), 1.0)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)


@pytest.mark.parametrize("axis", [None, (-2, -1), -1, ()], ids=lambda x: f"axis={x}")
def test_minmax_scaler(axis: Axis) -> None:
    r"""Test the MinMaxScaler."""
    TRUE_SHAPES: dict[Axis, tuple[int, ...]] = {
        None: (2, 3, 4, 5),
        (-2, -1): (4, 5),
        -1: (5,),
        (): (),
    }
    expected = TRUE_SHAPES[axis]

    X = RNG.normal(size=(2, 3, 4, 5))
    encoder = MinMaxScaler(axis=axis)
    assert_type(encoder, MinMaxScaler[Any])

    encoder.fit(X)
    result = encoder.params["ymin"].shape
    assert result == expected
    encoded = encoder.encode(X)
    assert encoded.min() >= 0.0
    assert encoded.max() <= 1.0
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
