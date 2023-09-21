r"""Generators for synthetic time series dataset.

Contrary to `tsdm.dataset`, which contains static, real-world dataset, this module
contains generators for synthetic dataset. By design each generator consists of

- Configuration parameters, e.g. number of dimensions etc.
- Allows sampling from the data ground truth distribution p(x,y)
- Allows estimating the Bayes Error, i.e. the best performance possible on the dataset.
"""

__all__ = [
    # Protocols
    "Generator",
    "TimeSeriesGenerator",
    "Distribution",
    "TimeSeriesDistribution",
    "IVP_Generator",
    "IVP_Solver",
]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import scipy.stats
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from tsdm.types.aliases import SizeLike
from tsdm.types.variables import any_co as T_co


@runtime_checkable
class Generator(Protocol[T_co]):
    r"""Protocol for generators."""

    @abstractmethod
    def rvs(self, size: SizeLike = ()) -> T_co:
        """Random variates of given type."""
        ...


@runtime_checkable
class TimeSeriesGenerator(Protocol[T_co]):
    r"""Protocol for generators."""

    @abstractmethod
    def rvs(self, t: ArrayLike, size: SizeLike = ()) -> T_co:
        """Random variates of given type."""
        ...


@runtime_checkable
class _Distribution(Protocol[T_co]):
    """Protocol for distributions.

    We follow the design of `scipy.stats.rv_continuous` and `scipy.stats.rv_discrete`.
    """

    def stats(
        self, *, loc: ArrayLike = 0, scale: ArrayLike = 1, moments: str = "mvsk"
    ) -> tuple[T_co, ...]:
        """Some statistics of the given RV."""

    def entropy(self, /) -> T_co:
        """Differential entropy of the RV."""
        raise NotImplementedError

    def moment(self, order: int) -> T_co:
        """Non-central moment of order n."""
        raise NotImplementedError

    def pdf(self, x: ArrayLike, /) -> T_co:
        """Probability density function at x of the given RV."""
        raise NotImplementedError

    def cdf(self, x: ArrayLike, /) -> T_co:
        """Cumulative distribution function of the given RV."""
        raise NotImplementedError

    def ppf(self, q: ArrayLike, /) -> T_co:
        """Percent point function (inverse of `cdf`) at q of the given RV."""
        raise NotImplementedError

    def sf(self, x: ArrayLike, /) -> T_co:
        """Survival function (1 - `cdf`) at x of the given RV."""
        raise NotImplementedError

    def isf(self, q: ArrayLike, /) -> T_co:
        """Inverse survival function at q of the given RV."""
        raise NotImplementedError

    def logpdf(self, x: ArrayLike, /) -> T_co:
        """Log of the probability density function at x of the given RV."""
        try:
            return self.pdf(x).log()
        except AttributeError as exc:
            raise NotImplementedError from exc

    def logcdf(self, x: ArrayLike, /) -> T_co:
        """Log of the cumulative distribution function at x of the given RV."""
        try:
            return self.cdf(x).log()
        except AttributeError as exc:
            raise NotImplementedError from exc

    def logsf(self, x: ArrayLike, /) -> T_co:
        """Log of the survival function of the given RV."""
        try:
            return self.sf(x).log()
        except AttributeError as exc:
            raise NotImplementedError from exc


@runtime_checkable
class Distribution(_Distribution[T_co], Generator[T_co], Protocol[T_co]):
    """Protocol for distributions."""


@runtime_checkable
class TimeSeriesDistribution(
    _Distribution[T_co], TimeSeriesGenerator[T_co], Protocol[T_co]
):
    """Protocol for time-series distributions."""


@runtime_checkable
class IVP_Solver(Protocol[T_co]):
    """Protocol for initial value problem solvers.

    This is desined to be compatible with several solvers from different libraries:

    Examples:
        - `scipy.integrate.odeint`
        - `torchdiffeq.odeint`
        - `torchsde.sdeint`

    Note:
        - `scipy.integrate.solve_ivp` has y0 before t, therefore, we require that
          y0 is passed as a keyword argument.
    """

    def __call__(self, system: Any, t: ArrayLike, /, *, y0: ArrayLike) -> T_co:
        """Solve the initial value problem."""
        ...


@runtime_checkable
class IVP_Generator(TimeSeriesGenerator[T_co], Protocol[T_co]):
    """Protocol for Initial Value Problems.

    Subsumes ODE-Generators and SDE-Generators.
    """

    @property
    def ivp_solver(self) -> IVP_Solver[T_co]:
        """Initial value problem solver."""
        return NotImplemented

    @property
    def system(self) -> Any:
        """System of differential equations."""
        return NotImplemented

    @abstractmethod
    def get_initial_state(self, size: SizeLike = ()) -> T_co:
        """Generate (multiple) initial state(s) y₀."""
        ...

    @abstractmethod
    def make_observations(self, sol: Any, /) -> T_co:
        """Create observations from the solution."""
        ...

    def solve_ivp(self, t: ArrayLike, /, *, y0: ArrayLike) -> T_co:
        """Solve the initial value problem."""
        if self.ivp_solver is NotImplemented or self.system is NotImplemented:
            raise NotImplementedError
        return self.ivp_solver(self.system, t, y0=y0)

    def rvs(self, t: ArrayLike, size: SizeLike = ()) -> T_co:
        """Random variates of given type."""
        # get the initial state
        y0 = self.get_initial_state(size=size)

        # solve the initial value problem
        sol = self.solve_ivp(t, y0=y0)

        # add observation noise
        observations = self.make_observations(sol)

        return observations


if TYPE_CHECKING:
    scipy_dist: type[Distribution] = scipy.stats.rv_continuous
    scipy_solver: IVP_Solver = solve_ivp
