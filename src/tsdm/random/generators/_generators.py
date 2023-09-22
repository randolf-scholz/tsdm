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
    # functions
    "solve_ivp",
]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike

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
        raise NotImplementedError

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
class ODE(Protocol[T_co]):
    """Represents a system of ordinary differential equations."""

    def __call__(self, t: ArrayLike, y: ArrayLike) -> T_co:
        """Evaluate the vector field at given time and state.

        .. Signature:: ``[(N,), (..., N, *D) -> (..., N, *D)``

        Args:
            t: list of time stamps
            y: list of states at time t

        Returns:
            f(t, y(t)) value of the veector field at time t and state y(t)
        """
        ...


@runtime_checkable
class IVP_Solver(Protocol[T_co]):
    """Protocol for initial value problem solvers.

    This is desined to be compatible with several solvers from different libraries:

    Examples:
        - `scipy.integrate.odeint`
            - expects system to be Callable[[t, y], ...] or Callable[[t, y], ...]
        - `scipy.integrate.solve_ivp`
            - expects system to be Callable[[t, y], ...]
        - `torchdiffeq.odeint`
            - expects system to be Callable[[t, y], ...]
        - `torchsde.sdeint`
            - expects system to be SDE object with methods
                - `f(self, t, y) -> ...` (drift)
                - `g(self, t, y) -> ...` (diffusion)

    Note:
        - `scipy.integrate.solve_ivp` has y0 before t, therefore, we require that
          y0 is passed as a keyword argument.
    """

    def __call__(self, system: ODE | Any, t: ArrayLike, /, *, y0: ArrayLike) -> T_co:
        """Solve the initial value problem.

        .. Signature:: ``[(N,), (..., *D) -> (..., N, *D)``

        Args:
            system: Some object that represents the dynamics of the systems.
            t: sorted list of N time points at which to solve for y.
            y0: Initial state at t[0], tensor of shape `(..., *D)`.

        Returns:
            array-like object y[t_i] containing the solution of the initial value problem.
        """
        ...


def solve_ivp(
    system: ODE, t: ArrayLike, /, *, y0: ArrayLike, **kwargs: Any
) -> np.ndarray:
    """Wrapped version of `scipy.integrate.solve_ivp` that matches the IVP_solver Protocol."""
    t_eval = np.asarray(t)
    t0 = t_eval.min()
    tf = t_eval.max()
    return scipy.integrate.solve_ivp(system, (t0, tf), y0=y0, t_eval=t_eval, **kwargs)


@runtime_checkable
class IVP_Generator(TimeSeriesGenerator[T_co], Protocol[T_co]):
    """Protocol for Generators that solve Initial Value Problems (IVP).

    Subsumes ODE-Generators and SDE-Generators.
    """

    @property
    def ivp_solver(self) -> IVP_Solver[T_co]:
        """Initial value problem solver."""
        return cast(IVP_Solver[T_co], solve_ivp)

    @property
    def system(self) -> Any | ODE:
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
        if self.ivp_solver is scipy.integrate.solve_ivp:
            raise ValueError(
                "scipy.integrate.solve_ivp does not match the IVP_solver Protocol,"
                " since it requires separate bounds [t0,tf] and evaluation points"
                " t_eval. Please use the wrapped version"
                " tsdm.random.generators.solve_ivp instead."
            )
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
    scipy_solver: IVP_Solver = scipy.integrate.solve_ivp
    _solve_ivp: IVP_Solver = solve_ivp
