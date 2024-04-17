"""Test sklearn encoders."""

import numpy as np
import pytest
from pytest import mark
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator

from tsdm.encoders.sklearn import (
    SKLEARN_ENCODERS,
    SKLEARN_TRANSFORMS,
    SklearnEncoder,
    SklearnTransform,
)

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


@mark.parametrize("name", SKLEARN_TRANSFORMS)
def test_transform(name: str) -> None:
    cls = SKLEARN_TRANSFORMS[name]
    assert issubclass(cls, SklearnTransform)
    assert issubclass(cls, BaseEstimator)
    assert issubclass(cls, TransformerMixin)
    assert name in SKLEARN_ENCODERS or not issubclass(cls, SklearnEncoder)
    encoder = cls()
    check_estimator(encoder)


@mark.parametrize("name", SKLEARN_ENCODERS)
def test_encoder(name: str) -> None:
    cls = SKLEARN_ENCODERS[name]
    assert issubclass(cls, SklearnEncoder)
    assert issubclass(cls, BaseEstimator)
    assert issubclass(cls, TransformerMixin)
    encoder = cls()
    check_estimator(encoder)


@mark.parametrize("name", SKLEARN_ENCODERS)
def test_left_inverse(name: str) -> None:
    if name == "KBinsDiscretizer":
        pytest.xfail("KBinsDiscretizer is not left-invertible")
    cls = SKLEARN_ENCODERS[name]
    x = np.array(SAMPLE_DATA[name])
    # x = np.random.randn(10, 1)
    encoder = cls()
    encoder.fit(x)
    z = encoder.transform(x)
    x_hat = encoder.inverse_transform(z)
    assert (x == x_hat).all() or np.allclose(x, x_hat)


def test_shared_attrs():
    shared_attrs = set.intersection(
        *(set(dir(cls)) for cls in SKLEARN_ENCODERS.values())
    )
    shared_classes = set.intersection(
        *(set(cls.__mro__) for cls in SKLEARN_ENCODERS.values())
    )
    print(shared_attrs)
    print(shared_classes)
