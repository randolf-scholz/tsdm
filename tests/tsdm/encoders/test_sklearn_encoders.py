"""Test sklearn encoders."""

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


def test_all_checked() -> None:
    assert (
        _SKLEARN_NOT_ENCODERS.keys() | SKLEARN_ENCODERS.keys()
        == SKLEARN_TRANSFORMS.keys()
    )


@mark.parametrize("encoder", SKLEARN_TRANSFORMS.values(), ids=SKLEARN_TRANSFORMS)
def test_transform(encoder: type[Transform]) -> None:
    assert issubclass(encoder, Transform)


@mark.parametrize("encoder", SKLEARN_ENCODERS.values(), ids=SKLEARN_ENCODERS)
def test_encoder(encoder: type[InvertibleTransform]) -> None:
    assert issubclass(encoder, InvertibleTransform)


@mark.parametrize("encoder", _SKLEARN_NOT_ENCODERS.values(), ids=_SKLEARN_NOT_ENCODERS)
def test_not_encoder(encoder: type[Transform]) -> None:
    assert not issubclass(encoder, InvertibleTransform)
