"""Encoders based on scikit-learn transformers."""

__all__ = [
    # Constants
    "SKLEARN_TRANSFORMS",
    "SKLEARN_ENCODERS",
    # ABCs & Protocols
    "Transform",
    "InvertibleTransform",
]

from sklearn import preprocessing as sk_preprocessing

from tsdm.encoders.base import InvertibleTransform, Transform

SKLEARN_TRANSFORMS: dict[str, type[Transform]] = {
    "Binarizer"           : sk_preprocessing.Binarizer,
    "FunctionTransformer" : sk_preprocessing.FunctionTransformer,
    "KBinsDiscretizer"    : sk_preprocessing.KBinsDiscretizer,
    "KernelCenterer"      : sk_preprocessing.KernelCenterer,
    "LabelBinarizer"      : sk_preprocessing.LabelBinarizer,
    "LabelEncoder"        : sk_preprocessing.LabelEncoder,
    "MaxAbsScaler"        : sk_preprocessing.MaxAbsScaler,
    "MinMaxScaler"        : sk_preprocessing.MinMaxScaler,
    "MultiLabelBinarizer" : sk_preprocessing.MultiLabelBinarizer,
    "Normalizer"          : sk_preprocessing.Normalizer,
    "OneHotEncoder"       : sk_preprocessing.OneHotEncoder,
    "OrdinalEncoder"      : sk_preprocessing.OrdinalEncoder,
    "PolynomialFeatures"  : sk_preprocessing.PolynomialFeatures,
    "PowerTransformer"    : sk_preprocessing.PowerTransformer,
    "QuantileTransformer" : sk_preprocessing.QuantileTransformer,
    "RobustScaler"        : sk_preprocessing.RobustScaler,
    "SplineTransformer"   : sk_preprocessing.SplineTransformer,
    "StandardScaler"      : sk_preprocessing.StandardScaler,
}  # fmt: skip
r"""Dictionary of all available sklearn transforms."""


SKLEARN_ENCODERS: dict[str, type[InvertibleTransform]] = {
    # "Binarizer"           : sk_preprocessing.Binarizer,
    "FunctionTransformer" : sk_preprocessing.FunctionTransformer,
    # "KBinsDiscretizer"    : sk_preprocessing.KBinsDiscretizer,  # NOTE: Not left-invertible!
    # "KernelCenterer"      : sk_preprocessing.KernelCenterer,
    "LabelBinarizer"      : sk_preprocessing.LabelBinarizer,
    "LabelEncoder"        : sk_preprocessing.LabelEncoder,
    "MaxAbsScaler"        : sk_preprocessing.MaxAbsScaler,
    "MinMaxScaler"        : sk_preprocessing.MinMaxScaler,
    "MultiLabelBinarizer" : sk_preprocessing.MultiLabelBinarizer,
    # "Normalizer"          : sk_preprocessing.Normalizer,
    "OneHotEncoder"       : sk_preprocessing.OneHotEncoder,
    "OrdinalEncoder"      : sk_preprocessing.OrdinalEncoder,
    # "PolynomialFeatures"  : sk_preprocessing.PolynomialFeatures,
    "PowerTransformer"    : sk_preprocessing.PowerTransformer,
    "QuantileTransformer" : sk_preprocessing.QuantileTransformer,
    "RobustScaler"        : sk_preprocessing.RobustScaler,
    # "SplineTransformer"   : sk_preprocessing.SplineTransformer,
    "StandardScaler"      : sk_preprocessing.StandardScaler,
}  # fmt: skip
r"""Dictionary of all available sklearn encoders."""
