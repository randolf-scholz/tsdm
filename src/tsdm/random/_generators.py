r"""Generators for synthetic time series dataset.

Contrary to `tsdm.dataset`, which contains static, real-world dataset, this module
contains generators for synthetic dataset. By design each generator consists of

- Configuration parameters, e.g. number of dimensions etc.
- Allows sampling from the data ground truth distribution p(x,y)
- Allows estimating the Bayes Error, i.e. the best performance possible on the dataset.
"""

__all__ = [
    # Constants
    "Generator",
    "GENERATORS",
]

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.stats import rv_continuous

from tsdm.types.aliases import SizeLike
from tsdm.types.protocols import Array
from tsdm.types.variables import any_var as T


@runtime_checkable
class Generator(Protocol):
    r"""Protocol for generators."""

    @abstractmethod
    def rvs(self, size: SizeLike, *, random_state: Optional = None):
        """Random variates of given type."""
        ...


@runtime_checkable
class Distribution(Generator):
    """Protocol for distributions.

    We follow the design of `scipy.stats.rv_continuous` and `scipy.stats.rv_discrete`.
    """

    def stats(self, *, loc: ArrayLike = 0, scale: ArrayLike = 1, moments: str = "mvsk"):
        """Some statistics of the given RV."""

    def entropy(self, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Differential entropy of the RV."""
        ...

    def moment(self, order: int, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Non-central moment of order n."""
        ...

    def pdf(self, x: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Probability density function at x of the given RV."""
        ...

    def cdf(self, x: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Cumulative distribution function of the given RV."""
        ...

    def ppf(self, q: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Percent point function (inverse of `cdf`) at q of the given RV."""
        ...

    def sf(self, x: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Survival function (1 - `cdf`) at x of the given RV."""
        ...

    def isf(self, q: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Inverse survival function at q of the given RV."""
        ...

    def logpdf(self, x: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Log of the probability density function at x of the given RV."""
        ...

    def logcdf(self, x: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Log of the cumulative distribution function at x of the given RV."""
        ...

    def logsf(self, x: ArrayLike, /, *, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Log of the survival function of the given RV."""
        ...


class IVP_Solver(Protocol[T]):
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

    def __call__(self, system: Any, t: T, *, y0: T) -> T:
        """Solve the initial value problem."""
        ...


class DiffEqGenerator(Generator[T]):
    """Protocol for Differential Equation Generators."""

    @property
    def ivp_solver(self) -> IVP_Solver:
        """Initial value problem solver."""
        ...

    @abstractmethod
    def get_initial_state(self) -> T:
        """Create initial state y₀."""
        ...

    @abstractmethod
    def make_observations(self, sol: Any) -> T:
        """Create observations from the solution."""
        ...

    def __call__(self, t: T, x: T) -> T:
        """Vector field f(t, x(t))."""
        ...

    def rvs(self, t: T, *, random_state=None):
        """Random variates of given type."""
        ...


class BaseODEGenerator(DiffEqGenerator):
    """Base class for ODE-Generators."""

    ivp_solver: IVP_Solver = solve_ivp

    def rvs(self, t, *, random_state=None):
        """Random variates of given type."""
        # get the initial state
        x0 = self.get_initial_state()

        # solve the initial value problem
        sol = self.ivp_solver(t, x0)

        # add observation noise
        observations = self.make_observations(sol)

        return observations


@dataclass
class DampenedPendulum(BaseODEGenerator):
    """Dampened Pendulum Simulation.

    The dampended pendulum is an autonomous system with two degrees of freedom.
    """

    _: KW_ONLY
    g: float = 9.81
    """Gravitational acceleration."""
    length: float = 1.0
    """Length of the pendulum."""
    mass: float = 1.0
    """Mass of the pendulum."""
    gamma: float = 0.25
    """Damping coefficient."""
    observation_noise: Distribution = stats.norm(loc=0, scale=0.05)
    """Noise distribution."""
    parameter_noise: Distribution = stats.norm(loc=0, scale=1)
    """Noise distribution."""

    def rvs(self, t, *, theta0: float = np.pi, omega0: float = 4.0, random_state=None):
        """Random variates of the dampened pendulum."""
        # add noise to the parameters
        theta0 += self.parameter_noise.rvs().clip(-2, +2)
        omega0 *= self.parameter_noise.rvs().clip(-2, +2)
        x0 = np.array([theta0, omega0])

        # simulate the pendulum
        values = solve_ivp(self.vector_field, t, x0)
        # add noise
        values += self.observation_noise.rvs(size=values.shape)
        return values

    def vector_field(self, state, t=None):
        """Vector field of the pendulum.

        .. Signature:: ``[(...,), (..., 2) -> (..., 2)``
        """
        assert t is None, "Time-dependency not supported."

        theta, omega = state[..., 0], state[..., 1]
        return np.stack(
            [
                omega,
                -(self.g / self.length) * np.sin(theta)
                - self.gamma / self.mass * omega,
            ]
        )


GENERATORS: dict[str, Generator] = {}
r"""Dictionary of all available generators."""
