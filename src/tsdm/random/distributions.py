r"""Probability distributions."""

__all__ = [
    # ABCs & Protocols
    "RV",
    "TimeSeriesRV",
    "Distribution",
    # Classes
    "Dirichlet",
]

from abc import abstractmethod
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from tsdm.constants import RNG
from tsdm.types.aliases import Size


@runtime_checkable
class RV[T](Protocol):  # +T
    r"""Protocol for random variable."""

    @abstractmethod
    def rvs(
        self, size: Size = (), *, random_state: Optional[int | Generator] = None
    ) -> T:
        r"""Random variates of the given type."""
        ...


@runtime_checkable
class TimeSeriesRV[T](Protocol):  # +T
    r"""Protocol for time series random variable $p(x∣t)$."""

    @abstractmethod
    def rvs(
        self,
        t: ArrayLike,
        size: Size = (),
        *,
        random_state: Optional[int | Generator] = None,
    ) -> T:
        r"""Generate random time series."""
        ...


@runtime_checkable
class Distribution[T](RV[T], Protocol):  # +T
    r"""Protocol for distributions.

    We follow the design of `scipy.stats.rv_continuous` and `scipy.stats.rv_discrete`.
    """

    def stats(
        self, *, loc: ArrayLike = 0, scale: ArrayLike = 1, moments: str = "mvsk"
    ) -> tuple[T, ...]:
        r"""Some statistics of the given RV."""
        raise NotImplementedError

    def entropy(self, /) -> T:
        r"""Differential entropy of the RV."""
        raise NotImplementedError

    def moment(self, order: int) -> T:
        r"""Non-central moment of order n."""
        raise NotImplementedError

    def pdf(self, x: ArrayLike, /) -> T:
        r"""Probability density function at x of the given RV."""
        raise NotImplementedError

    def cdf(self, x: ArrayLike, /) -> T:
        r"""Cumulative distribution function of the given RV."""
        raise NotImplementedError

    def ppf(self, q: ArrayLike, /) -> T:
        r"""Percent point function (inverse of `cdf`) at q of the given RV."""
        raise NotImplementedError

    def sf(self, x: ArrayLike, /) -> T:
        r"""Survival function (1 - `cdf`) at x of the given RV."""
        raise NotImplementedError

    def isf(self, q: ArrayLike, /) -> T:
        r"""Inverse survival function at q of the given RV."""
        raise NotImplementedError

    def logpdf(self, x: ArrayLike, /) -> T:
        r"""Log of the probability density function at x of the given RV."""
        try:
            return self.pdf(x).log()  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise NotImplementedError from exc

    def logcdf(self, x: ArrayLike, /) -> T:
        r"""Log of the cumulative distribution function at x of the given RV."""
        try:
            return self.cdf(x).log()  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise NotImplementedError from exc

    def logsf(self, x: ArrayLike, /) -> T:
        r"""Log of the survival function of the given RV."""
        try:
            return self.sf(x).log()  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise NotImplementedError from exc


class Dirichlet:
    r"""Vectorized version of `scipy.stats.dirichlet`."""

    def __init__(self) -> None:
        r"""Initialize the Dirichlet distribution."""
        raise NotImplementedError

    @classmethod
    def rvs(
        cls,
        alphas: ArrayLike,
        size: Size = (),
        *,
        random_state: Optional[int | Generator] = None,
    ) -> NDArray:
        r"""Random variates of the Dirichlet distribution."""
        alphas = np.asarray(alphas)
        size = (size,) if isinstance(size, int) else size
        rng = RNG if random_state is None else np.random.default_rng(random_state)
        x = rng.gamma(shape=alphas, size=size + alphas.shape)
        x /= x.sum(axis=-1, keepdims=True)
        return x
