r"""Test compatibility with sklearn encoders."""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks


class MyEstimator:
    r"""Dummy estimator."""

    def get_params(self, *, deep=True):
        return {}

    def set_params(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        return X

    def __call__(self, X):
        return self.transform(X)


@parametrize_with_checks([MinMaxScaler(), StandardScaler()])
def test_sklearn_compatibility(estimator, check):
    check(estimator)
