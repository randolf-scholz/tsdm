r"""Test numerical encoders."""

import logging
import pickle
from collections.abc import Callable
from tempfile import TemporaryFile

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray
from typing_extensions import Any, TypeAlias, TypeVar, assert_type

from tsdm.constants import RNG
from tsdm.encoders.numerical import (
    BoundaryEncoder,
    ClippingMode,
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
Bounds: TypeAlias = tuple[float | None, float | None]


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

DATA = {
    "numpy-1D": np.array(DATA_1D),
    "numpy-2D": np.array(DATA_2D),
    "torch-1D": torch.tensor(DATA_1D),
    "torch-2D": torch.tensor(DATA_2D),
    "pandas-I": pd.Index(DATA_1D),
    "pandas-S": pd.Series(DATA_1D),
    # "pandas-dataframe": pd.DataFrame(_DATA_2D),
}
BOUNDS: list[Bounds] = [
    (-1, +1),
    (0, 1),
    (0, float("inf")),
    (0, None),
    (0, float("nan")),
    (0, pd.NA),
]


@pytest.mark.parametrize("upper_included", [True, False])
@pytest.mark.parametrize("lower_included", [True, False])
@pytest.mark.parametrize("bounds", BOUNDS, ids=str)
@pytest.mark.parametrize("mode", ["clip", "mask"])
@pytest.mark.parametrize("data", DATA.values(), ids=DATA)
def test_boundary_encoder2(
    *,
    data: D,
    mode: ClippingMode,
    bounds: Bounds,
    lower_included: bool,
    upper_included: bool,
) -> None:
    r"""Test the boundary encoder."""
    # create the encoder
    encoder = BoundaryEncoder(
        bounds[0],
        bounds[1],
        mode=mode,
        lower_included=lower_included,
        upper_included=upper_included,
        axis=-1,
    )

    # fit the encoder
    encoder.fit(data)

    # compute the encoded data
    encoded = encoder.encode(data)
    assert type(encoded) == type(data)
    assert encoded.shape == data.shape
    assert encoded.dtype == data.dtype

    lb, ub = encoder.lower_bound, encoder.upper_bound
    nan_data = np.isnan(data)
    nan_encoded = np.isnan(encoded)

    match pd.isna(lb), mode, lower_included:
        case True, _, _:
            lower_mask = np.zeros_like(data, dtype=bool)
        case False, "clip", _:
            lower_mask = data <= lb
        case False, "mask", True:
            lower_mask = data < lb
        case False, "mask", False:
            lower_mask = data <= lb
        case _:
            raise ValueError(f"Unexpected combination: {lb=} {mode=} {lower_included=}")

    match pd.isna(ub), mode, upper_included:
        case True, _, _:
            upper_mask = np.zeros_like(data, dtype=bool)
        case False, "clip", _:
            upper_mask = data >= ub
        case False, "mask", True:
            upper_mask = data > ub
        case False, "mask", False:
            upper_mask = data >= ub
        case _:
            raise ValueError(f"Unexpected combination: {ub=} {mode=} {upper_included=}")

    match mode:
        case "clip":
            assert (nan_data == nan_encoded).all()
            assert ((encoded == ub) == upper_mask).all()
            assert ((encoded == lb) == lower_mask).all()
        case "mask":
            assert (nan_encoded == (nan_data | lower_mask | upper_mask)).all()
        case _:
            raise ValueError(f"Unexpected mode: {mode=}")


@pytest.mark.parametrize("data", DATA.values(), ids=DATA)
def test_boundary_encoder(data: D) -> None:
    r"""Test the boundary encoder."""
    encoder = BoundaryEncoder(-1, +1, mode="clip", axis=-1)
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert encoded.shape == data.shape
    assert ((encoded >= -1) & (encoded <= 1)).all()
    assert (encoded == -1).sum() == (data <= -1).sum()
    assert (encoded == +1).sum() == (data >= +1).sum()

    match data:
        case np.ndarray() as array:
            assert isinstance(encoded, np.ndarray)
            assert encoded.dtype == array.dtype
        case torch.Tensor() as tensor:
            assert isinstance(encoded, torch.Tensor)
            assert encoded.device == tensor.device
            assert encoded.dtype == tensor.dtype
        case pd.Index() as index:
            assert isinstance(encoded, pd.Index)
            assert encoded.name == index.name
        case pd.Series() as series:
            assert isinstance(encoded, pd.Series)
            assert encoded.name == series.name
            assert encoded.index.equals(series.index)
        case pd.DataFrame() as df:
            assert isinstance(encoded, pd.DataFrame)
            assert encoded.columns.equals(df.columns)
            assert encoded.index.equals(df.index)
        case _:
            raise TypeError(f"Unexpected type: {type(data)}")

    # test numpy + mask
    encoder = BoundaryEncoder(-1, +1, mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert (np.isnan(encoded) ^ ((encoded >= -1) & (encoded <= 1))).all()
    assert np.isnan(encoded).sum() == ((data < -1).sum() + (data > +1).sum())

    # test fitting with mask
    encoder = BoundaryEncoder(mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    assert not (np.isnan(encoded)).any()
    assert (data == encoded).all()
    assert (data == decoded).all()

    # encode some data that violates bounds
    data2 = data * 2
    encoded2 = encoder.encode(data2)
    xmin, xmax = data.min(), data.max()
    mask = (data2 >= xmin) & (data2 <= xmax)
    assert (encoded2[mask] == data2[mask]).all()
    assert (np.isnan(encoded2[~mask])).all()

    # test half-open interval + clip
    encoder = BoundaryEncoder(0, None, mode="clip")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert (encoded >= 0).all()
    assert (encoded == 0).sum() == (data <= 0).sum()

    # test half-open unbounded interval + mask
    encoder = BoundaryEncoder(0, None, mode="mask")
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert (np.isnan(encoded) ^ (encoded >= 0)).all()
    assert np.isnan(encoded).sum() == (data < 0).sum()

    # test half-open bounded interval + mask
    encoder = BoundaryEncoder(0, 1, mode="mask", lower_included=False)
    encoder.fit(data)
    encoded = encoder.encode(data)
    assert (np.isnan(encoded) ^ (encoded > 0)).all()
    assert np.isnan(encoded).sum() == ((data <= 0).sum() + (data > 1).sum())


@pytest.mark.parametrize(
    "tensor_type", [pd.Series, pd.DataFrame, np.array, torch.tensor]
)
def test_linear_scaler(tensor_type: T) -> None:
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
    assert isinstance(encoder.params["mean"], float), f"{encoder.params}"

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
    assert isinstance(encoder.params["mean"], float), f"{encoder.params}"

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
    assert isinstance(encoder.params["mean"], float), f"{encoder.params}"


@pytest.mark.parametrize("shape", [(5, 2, 3, 4), (7,)], ids=str)
@pytest.mark.parametrize("axis", [(1, -1), (-1,), -2, 0, None, ()], ids=str)
def test_get_broadcast(
    shape: tuple[int, ...], axis: None | int | tuple[int, ...]
) -> None:
    r"""Test the get_broadcast function."""
    if (isinstance(axis, tuple) and any(abs(a) > len(shape) - 1 for a in axis)) or (
        isinstance(axis, int) and abs(axis) > len(shape)
    ):
        pytest.skip(f"{shape=} {axis=}")

    arr: NDArray = RNG.normal(size=shape)

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
    r"""Test the get_reduced_axes function."""
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


@pytest.mark.parametrize("encoder_type", [StandardScaler, MinMaxScaler])
@pytest.mark.parametrize(
    "tensor_type", [pd.Series, pd.DataFrame, np.array, torch.tensor]
)
def test_scaler(encoder_type: type[E], tensor_type: T) -> None:
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
        assert encoder.params["mean"].shape == (), f"{encoder.params}"

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
    assert encoder.params["mean"].shape == (5,), f"{encoder.params}"

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
    assert encoder.params["mean"].shape == ()

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
    assert encoder.params["mean"].shape == (4, 5), f"{encoder.params}"

    LOGGER.info("Testing finished!")


@pytest.mark.parametrize("encoder_type", [StandardScaler, MinMaxScaler])
def test_scaler_dataframe(encoder_type: type[E]) -> None:
    r"""Check whether the scaler-encoders work as expected on DataFrame."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing Encoder applied to pandas.DataFrame.")

    LOGGER.info("Testing without batching.")
    X = pd.DataFrame(RNG.uniform(size=(5, 3)), columns=["a", "b", "c"])
    encoder = encoder_type(axis=-1)

    # validate fitting
    encoder.fit(X)
    assert encoder.params["mean"].shape == (3,), f"{encoder.params}"

    # validate encoding
    encoded = encoder.encode(X)
    assert isinstance(encoded, pd.DataFrame)
    assert encoded.shape == X.shape
    assert encoded.columns.equals(X.columns)
    assert encoded.index.equals(X.index)

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


@pytest.mark.parametrize("encoder_type", [StandardScaler, MinMaxScaler])
def test_scaler_series(encoder_type: type[E]) -> None:
    r"""Check whether the scaler-encoders work as expected on Series."""
    LOGGER = __logger__.getChild(encoder_type.__name__)
    LOGGER.info("Testing Encoder applied to pandas.Series.")

    X = pd.Series([-1.0, 1.2, 2.7, 3.0], name="foo")
    encoder = encoder_type()

    # validate fitting
    encoder.fit(X)
    assert encoder.params["mean"].shape == (), f"{encoder.params}"

    # validate encoding
    encoded = encoder.encode(X)
    assert isinstance(encoded, pd.Series)
    assert encoded.shape == X.shape
    assert encoded.name == X.name
    assert encoded.index.equals(X.index)

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


@pytest.mark.parametrize("axis", [None, (-2, -1), -1, ()], ids=lambda x: f"axis={x}")
def test_standard_scaler(axis):
    r"""Test the MinMaxScaler."""
    TRUE_SHAPE = {
        (): (),
        (-2, -1): (4, 5),
        -1: (5,),
        None: (2, 3, 4, 5),
    }[axis]

    # initialize
    X = RNG.normal(size=(2, 3, 4, 5))
    encoder = StandardScaler(axis=axis)
    assert_type(encoder, StandardScaler[Any])

    # fit to data
    encoder.fit(X)
    assert encoder.params["ymin"].shape == TRUE_SHAPE
    encoded = encoder.encode(X)

    if axis is None:  # std = 0.0
        return

    assert np.allclose(encoded.mean(), 0.0)
    assert np.allclose(encoded.std(), 1.0)
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)


@pytest.mark.parametrize("axis", [None, (-2, -1), -1, ()], ids=lambda x: f"axis={x}")
def test_minmax_scaler(axis):
    r"""Test the MinMaxScaler."""
    TRUE_SHAPE = {
        None: (2, 3, 4, 5),
        (-2, -1): (4, 5),
        -1: (5,),
        (): (),
    }[axis]

    X = RNG.normal(size=(2, 3, 4, 5))
    encoder = MinMaxScaler(axis=axis)
    assert_type(encoder, MinMaxScaler[Any])

    encoder.fit(X)
    assert encoder.params["ymin"].shape == TRUE_SHAPE
    encoded = encoder.encode(X)
    assert encoded.min() >= 0.0
    assert encoded.max() <= 1.0
    decoded = encoder.decode(encoded)
    assert np.allclose(X, decoded)
