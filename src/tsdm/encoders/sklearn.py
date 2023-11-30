#!/usr/bin/env python


from sklearn import preprocessing as sk_preprocessing
from sklearn.base import BaseEstimator
from typing_extensions import Protocol


class Transformer(Protocol):
    """Protocol for transformers."""

    def fit(self, X, y=None):
        """Fit the transformer."""
        ...

    def transform(self, X):
        """Transform the data."""
        ...

    def fit_transform(self, X, y=None):
        """Fit and transform the data."""
        ...


x: type[Transformer] = BaseEstimator
reveal_type(BaseEstimator)

SKLEARN_ENCODERS: dict[str, type[Transformer]] = {
    "Binarizer": sk_preprocessing.Binarizer,
    "FunctionTransformer": sk_preprocessing.FunctionTransformer,
    "KBinsDiscretizer": sk_preprocessing.KBinsDiscretizer,
    "KernelCenterer": sk_preprocessing.KernelCenterer,
    "LabelBinarizer": sk_preprocessing.LabelBinarizer,
    "LabelEncoder": sk_preprocessing.LabelEncoder,
    "MaxAbsScaler": sk_preprocessing.MaxAbsScaler,
    "MinMaxScaler": sk_preprocessing.MinMaxScaler,
    "MultiLabelBinarizer": sk_preprocessing.MultiLabelBinarizer,
    "Normalizer": sk_preprocessing.Normalizer,
    "OneHotEncoder": sk_preprocessing.OneHotEncoder,
    "OrdinalEncoder": sk_preprocessing.OrdinalEncoder,
    "PolynomialFeatures": sk_preprocessing.PolynomialFeatures,
    "PowerTransformer": sk_preprocessing.PowerTransformer,
    "QuantileTransformer": sk_preprocessing.QuantileTransformer,
    "RobustScaler": sk_preprocessing.RobustScaler,
    "SplineTransformer": sk_preprocessing.SplineTransformer,
    "StandardScaler": sk_preprocessing.StandardScaler,
}
r"""Dictionary of all available sklearn encoders."""
