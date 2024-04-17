"""Encoders based on scikit-learn transformers."""

__all__ = [
    # Constants
    "SKLEARN_TRANSFORMS",
    "SKLEARN_ENCODERS",
    # ABCs & Protocols
    "SklearnTransform",
    "SklearnEncoder",
]

from abc import abstractmethod

from sklearn import preprocessing as sk_preprocessing
from typing_extensions import Any, Protocol, runtime_checkable

from tsdm.types.variables import T2, T, T_co, T_contra


@runtime_checkable
class SklearnTransform(Protocol[T_contra, T_co]):
    """Protocol for scikit-learn transformers."""

    @abstractmethod
    def fit(self, data: T_contra, /) -> None: ...
    @abstractmethod
    def transform(self, data: T_contra, /) -> T_co: ...
    @abstractmethod
    def fit_transform(self, data: T_contra, /) -> T_co: ...
    @abstractmethod
    def get_params(self, *, deep: bool = True) -> dict[str, Any]: ...
    @abstractmethod
    def set_params(self, **params: Any) -> None: ...


@runtime_checkable
class SklearnEncoder(SklearnTransform[T, T2], Protocol[T, T2]):
    """Protocol for scikit-learn encoders."""

    @abstractmethod
    def inverse_transform(self, data: T2, /) -> T: ...


SKLEARN_TRANSFORMS: dict[str, type[SklearnTransform]] = {
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


SKLEARN_ENCODERS: dict[str, type[SklearnEncoder]] = {
    # "Binarizer"           : sk_preprocessing.Binarizer,
    "FunctionTransformer" : sk_preprocessing.FunctionTransformer,
    "KBinsDiscretizer"    : sk_preprocessing.KBinsDiscretizer,  # NOTE: Not left-invertible!
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
