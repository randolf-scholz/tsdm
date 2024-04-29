"""Probability distributions."""

__all__ = [
    # ABCs & Protocols
    "Generator",
    "TimeSeriesGenerator",
    "Distribution",
    "TimeSeriesDistribution",
    # Classes
    "Dirichlet",
]

from abc import abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Protocol, runtime_checkable

from tsdm.constants import RNG
from tsdm.types.aliases import Size
from tsdm.types.variables import T_co


@runtime_checkable
class Generator(Protocol[T_co]):
    r"""Protocol for generators."""

    @abstractmethod
    def rvs(self, size: Size = ()) -> T_co:
        r"""Random variates of the given type."""
        ...


@runtime_checkable
class TimeSeriesGenerator(Protocol[T_co]):
    r"""Protocol for generators."""

    @abstractmethod
    def rvs(self, t: ArrayLike, size: Size = ()) -> T_co:
        r"""Random variates of the given type."""
        ...


@runtime_checkable
class _Distribution(Protocol[T_co]):
    r"""Protocol for distributions.

    We follow the design of `scipy.stats.rv_continuous` and `scipy.stats.rv_discrete`.
    """

    def stats(
        self, *, loc: ArrayLike = 0, scale: ArrayLike = 1, moments: str = "mvsk"
    ) -> tuple[T_co, ...]:
        r"""Some statistics of the given RV."""
        raise NotImplementedError

    def entropy(self, /) -> T_co:
        r"""Differential entropy of the RV."""
        raise NotImplementedError

    def moment(self, order: int) -> T_co:
        r"""Non-central moment of order n."""
        raise NotImplementedError

    def pdf(self, x: ArrayLike, /) -> T_co:
        r"""Probability density function at x of the given RV."""
        raise NotImplementedError

    def cdf(self, x: ArrayLike, /) -> T_co:
        r"""Cumulative distribution function of the given RV."""
        raise NotImplementedError

    def ppf(self, q: ArrayLike, /) -> T_co:
        r"""Percent point function (inverse of `cdf`) at q of the given RV."""
        raise NotImplementedError

    def sf(self, x: ArrayLike, /) -> T_co:
        r"""Survival function (1 - `cdf`) at x of the given RV."""
        raise NotImplementedError

    def isf(self, q: ArrayLike, /) -> T_co:
        r"""Inverse survival function at q of the given RV."""
        raise NotImplementedError

    def logpdf(self, x: ArrayLike, /) -> T_co:
        r"""Log of the probability density function at x of the given RV."""
        try:
            return self.pdf(x).log()
        except AttributeError as exc:
            raise NotImplementedError from exc

    def logcdf(self, x: ArrayLike, /) -> T_co:
        r"""Log of the cumulative distribution function at x of the given RV."""
        try:
            return self.cdf(x).log()
        except AttributeError as exc:
            raise NotImplementedError from exc

    def logsf(self, x: ArrayLike, /) -> T_co:
        r"""Log of the survival function of the given RV."""
        try:
            return self.sf(x).log()
        except AttributeError as exc:
            raise NotImplementedError from exc


@runtime_checkable
class Distribution(_Distribution[T_co], Generator[T_co], Protocol[T_co]):
    r"""Protocol for distributions."""


@runtime_checkable
class TimeSeriesDistribution(
    _Distribution[T_co], TimeSeriesGenerator[T_co], Protocol[T_co]
):
    r"""Protocol for time-series distributions."""


class Dirichlet:
    r"""Vectorized version of `scipy.stats.dirichlet`."""

    def __init__(self):
        r"""Initialize the Dirichlet distribution."""
        raise NotImplementedError

    @classmethod
    def rvs(cls, alphas: ArrayLike, size: Size = ()) -> NDArray:
        r"""Random variates of the Dirichlet distribution."""
        alphas = np.asarray(alphas)
        size = (size,) if isinstance(size, int) else size
        x = RNG.gamma(shape=alphas, size=size + alphas.shape)
        x /= x.sum(axis=-1, keepdims=True)
        return x
