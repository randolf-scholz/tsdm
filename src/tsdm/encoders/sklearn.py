"""Encoders based on scikit-learn transformers."""

__all__ = [
    # Types
    "Transform",
    "InvertibleTransform",
    # Constants
    "SKLEARN_TRANSFORMS",
    "SKLEARN_ENCODERS",
]


from sklearn import preprocessing as sk_preprocessing
from typing_extensions import Protocol, TypeVar, runtime_checkable

U = TypeVar("U")
V = TypeVar("V")
U_contra = TypeVar("U_contra", contravariant=True)
V_co = TypeVar("V_co", covariant=True)


@runtime_checkable
class Transform(Protocol[U_contra, V_co]):
    """Protocol for transformers."""

    def fit(self, X: U_contra, /) -> None:
        """Fit the transformer."""
        ...

    def transform(self, X: U_contra, /) -> V_co:
        """Transform the data."""
        ...


@runtime_checkable
class InvertibleTransform(Transform[U, V], Protocol):
    """Protocol for invertible transformers."""

    def inverse_transform(self, X: V, /) -> U:
        """Reverse transform the data."""
        ...


SKLEARN_TRANSFORMS: dict[str, type[Transform]] = {
    # fmt: off
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
    # fmt: on
}
r"""Dictionary of all available sklearn transforms."""


SKLEARN_ENCODERS: dict[str, type[InvertibleTransform]] = {
    # fmt: off
    # "Binarizer"           : sk_preprocessing.Binarizer,
    "FunctionTransformer" : sk_preprocessing.FunctionTransformer,
    "KBinsDiscretizer"    : sk_preprocessing.KBinsDiscretizer,
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
    # fmt: on
}
r"""Dictionary of all available sklearn encoders."""
