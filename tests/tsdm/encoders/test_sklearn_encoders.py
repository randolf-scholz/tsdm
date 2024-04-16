"""Test sklearn encoders."""

import numpy as np
from pytest import mark
from sklearn import preprocessing as sk_preprocessing

from tsdm.encoders.sklearn import (
    SKLEARN_ENCODERS,
    SKLEARN_TRANSFORMS,
    InvertibleTransform,
    Transform,
)

_SKLEARN_NOT_ENCODERS = {
    "Binarizer": sk_preprocessing.Binarizer,
    "KernelCenterer": sk_preprocessing.KernelCenterer,
    "Normalizer": sk_preprocessing.Normalizer,
    "PolynomialFeatures": sk_preprocessing.PolynomialFeatures,
    "SplineTransformer": sk_preprocessing.SplineTransformer,
}

BINARY_DATA = np.array(["yes", "no", "no", "yes", "yes"])
CATEGORICAL_DATA = np.array([["car"], ["bike"], ["car"], ["bike"], ["house"]])
NUMERICAL_DATA = np.array([[-3.7], [-1.3], [0.0], [0.3], [2.1], [3.2]])
UNIVARIATE_CAT_DATA = np.array(["car", "bike", "car", "bike", "house"])
MULTICATEGORICAL_DATA = np.array([
    ["car", "red"],
    ["blue", "car"],
    ["bike", "green"],
    ["bike", "red"],
    ["car", "green"],
])

SAMPLE_DATA = {
    "FunctionTransformer" : NUMERICAL_DATA,
    "KBinsDiscretizer"    : NUMERICAL_DATA,
    "LabelBinarizer"      : BINARY_DATA,
    "LabelEncoder"        : UNIVARIATE_CAT_DATA,
    "MaxAbsScaler"        : NUMERICAL_DATA,
    "MinMaxScaler"        : NUMERICAL_DATA,
    "MultiLabelBinarizer" : MULTICATEGORICAL_DATA,
    "OneHotEncoder"       : CATEGORICAL_DATA,
    "OrdinalEncoder"      : CATEGORICAL_DATA,
    "PowerTransformer"    : NUMERICAL_DATA,
    "QuantileTransformer" : NUMERICAL_DATA,
    "RobustScaler"        : NUMERICAL_DATA,
    "StandardScaler"      : NUMERICAL_DATA,
}  # fmt: skip
r"""Dictionary of all available sklearn encoders."""


def test_all_checked() -> None:
    assert (
        _SKLEARN_NOT_ENCODERS.keys() | SKLEARN_ENCODERS.keys()
        == SKLEARN_TRANSFORMS.keys()
    )


@mark.parametrize("cls", SKLEARN_TRANSFORMS.values(), ids=SKLEARN_TRANSFORMS)
def test_transform(cls: type[Transform]) -> None:
    assert issubclass(cls, Transform)


@mark.parametrize("cls", SKLEARN_ENCODERS.values(), ids=SKLEARN_ENCODERS)
def test_encoder(cls: type[InvertibleTransform]) -> None:
    assert issubclass(cls, InvertibleTransform)


@mark.parametrize("name", SKLEARN_ENCODERS)
def test_left_inverse(name: str) -> None:
    cls = SKLEARN_ENCODERS[name]
    x = np.array(SAMPLE_DATA[name])
    # x = np.random.randn(10, 1)
    encoder = cls()
    encoder.fit(x)
    z = encoder.transform(x)
    x_hat = encoder.inverse_transform(z)
    assert (x == x_hat).all() or np.allclose(x, x_hat)


@mark.parametrize("cls", _SKLEARN_NOT_ENCODERS.values(), ids=_SKLEARN_NOT_ENCODERS)
def test_not_encoder(cls: type[Transform]) -> None:
    assert not issubclass(cls, InvertibleTransform)
