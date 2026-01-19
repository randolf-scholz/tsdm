r"""Encoders based on scikit-learn transformers."""

__all__ = [
    # Constants
    "SKLEARN_TRANSFORMS",
    "SKLEARN_ENCODERS",
    # ABCs & Protocols
    "InvertibleSklearnTransform",
    "SklearnEncoder",
    "SklearnTransform",
]

from abc import abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from sklearn import preprocessing as sk_preprocessing


# region sklearn protocols -------------------------------------------------------------
@runtime_checkable
class SklearnTransform[X, Y](Protocol):  # -X, +Y
    r"""Protocol for transformers."""

    @abstractmethod
    def fit(self, x: X, /) -> None: ...
    @abstractmethod
    def transform(self, x: X, /) -> Y: ...


@runtime_checkable
class InvertibleSklearnTransform[X, Y](SklearnTransform[X, Y], Protocol):
    r"""Protocol for invertible transformers."""

    @abstractmethod
    def inverse_transform(self, y: Y, /) -> X: ...


@runtime_checkable
class SklearnEncoder[X, Y](Protocol):
    r"""Protocol for scikit-learn transformers."""

    def transform(self, x: X, /) -> Y: ...
    def inverse_transform(self, x: Y, /) -> X: ...

    def fit(self, x: X, /, y: Optional[Y] = None) -> None: ...
    def fit_transform(self, x: X, y: Optional[Y] = None, **fit_params: Any) -> None: ...

    def get_params(self, *, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> None: ...
    def __setstate__(self, state: dict[str, Any], /) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...


# endregion sklearn protocols ----------------------------------------------------------


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
