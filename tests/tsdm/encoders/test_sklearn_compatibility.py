"""Test compatibility with sklearn encoders."""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

# from tsdm.encoders.numerical import MinMaxScaler, StandardScaler


class MyEstimator:
    """Dummy estimator."""

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


a = StandardScaler()
# a.__name__ = "StandardScaler"
b = MinMaxScaler()
# b.__name__ = "MinMaxScaler"
c = MyEstimator()


@parametrize_with_checks([c])
def test_sklearn_compatibility(estimator, check):
    check(estimator)
