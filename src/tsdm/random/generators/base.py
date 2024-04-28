r"""Generators for synthetic time series dataset.

Contrary to `tsdm.dataset`, which contains static, real-world dataset, this module
contains generators for synthetic dataset. By design, each generator consists of

- Configuration parameters, e.g. number of dimensions etc.
- Allows sampling from the data ground truth distribution $p(x,y)$.
- Allows estimating the Bayes Error, i.e. the best performance possible on the dataset.
"""

__all__ = [
    # ABCs & Protocols
    "IVP_Generator",
    "IVP_GeneratorBase",
    "IVP_Solver",
    "ODE",
    "TimeSeriesGenerator",
    # Functions
    "solve_ivp",
]

from abc import abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp as scipy_solve_ivp
from scipy.optimize import OptimizeResult as OdeResult
from scipy.stats import rv_continuous
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
    final,
    runtime_checkable,
)

from tsdm.random.stats.distributions import Distribution
from tsdm.types.aliases import SizeLike
from tsdm.types.callback_protocols import NullMap, SelfMap
from tsdm.types.variables import T_co


@runtime_checkable
class TimeSeriesGenerator(Protocol[T_co]):
    r"""Protocol for generators."""

    @abstractmethod
    def rvs(self, t: ArrayLike, size: SizeLike = ()) -> T_co:
        r"""Random variates of the given type."""
        ...


@runtime_checkable
class ODE(Protocol[T_co]):
    r"""Represents a system of ordinary differential equations."""

    @abstractmethod
    def __call__(self, t: ArrayLike, y: ArrayLike) -> T_co:
        r"""Evaluate the vector field at given time and state.

        .. signature:: ``[(N,), (..., N, *D) -> (..., N, *D)``

        Sub-signatures:
            - ``[(,), (..., *D) -> (..., *D)``
            - ``[(...,), (*D,) -> (..., *D)``
            - ``[(,), (*D,) -> (*D,)``

        Args:
            t: list of time stamps
            y: list of states at time t

        Returns:
            f(t, y(t)) value of the veector field at time t and state y(t)
        """
        ...


@runtime_checkable
class IVP_Solver(Protocol[T_co]):
    r"""Protocol for initial value problem solvers.

    This is designed to be compatible with several solvers from different libraries:

    Examples:
        - `scipy.integrate.odeint`
            - expects system-component to be `Callable[[t, y], ...]` or `Callable[[t, y], ...]`
        - `scipy.integrate.solve_ivp`
            - expects system to be `Callable[[t, y], ...]`
        - `torchdiffeq.odeint`
            - expects system to be `Callable[[t, y], ...]`
        - `torchsde.sdeint`
            - expects system to be SDE object with methods
                - `f(self, t, y) -> ...` (drift)
                - `g(self, t, y) -> ...` (diffusion)

    Note:
        - `scipy.integrate.solve_ivp` has y0 before t, therefore, we require that
          `y0` is passed as a keyword argument.
    """

    @abstractmethod
    def __call__(self, system: ODE | Any, t: ArrayLike, /, *, y0: ArrayLike) -> T_co:
        r"""Solve the initial value problem.

        .. signature:: ``[(N,), (..., *D) -> (..., N, *D)``

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
    r"""Wrapped version of `scipy.integrate.solve_ivp` that matches the IVP_solver Protocol."""
    t_eval = np.asarray(t)
    t0 = t_eval.min()
    tf = t_eval.max()
    sol: OdeResult = scipy_solve_ivp(system, (t0, tf), y0=y0, t_eval=t_eval, **kwargs)
    # NOTE: output shape: (d, n_timestamps), move time axis to the front
    return np.moveaxis(sol.y, -1, 0)


@runtime_checkable
class IVP_Generator(TimeSeriesGenerator[T_co], Protocol[T_co]):
    r"""Protocol for Generators that solve Initial Value Problems (IVP).

    Needs to implement the following things:

    - `get_initial_state` to generate initial states x₀
    - `solve_ivp` to solve the initial value problem

    Subsumes ODE-Generators and SDE-Generators.

    Examples:
        Nonlinear ODE

        .. math::
             \dot{x}(t) &= f(t, x(t)) \\
                   y(t) &= g(t, x(t))
    """

    @abstractmethod
    def get_initial_state(self, size: SizeLike = ()) -> T_co:
        r"""Generate (multiple) initial state(s) y₀."""
        ...

    @abstractmethod
    def make_observations(self, sol: Any, /) -> T_co:
        r"""Create observations from the solution."""
        ...

    @abstractmethod
    def solve_ivp(self, t: ArrayLike, /, *, y0: ArrayLike) -> Any:
        r"""Solve the initial value problem."""
        ...

    def rvs(self, t: ArrayLike, size: SizeLike = ()) -> T_co:
        r"""Random variates of the given type."""
        # get the initial state
        y0 = self.get_initial_state(size=size)

        # solve the initial value problem
        sol = self.solve_ivp(t, y0=y0)

        # get observations (add noise))
        obs = self.make_observations(sol)

        return obs


class IVP_GeneratorBase(IVP_Generator[T_co]):
    r"""Base class for IVP_Generators."""

    @property
    def system(self) -> ODE | Any:
        r"""System of differential equations."""
        return NotImplemented

    @property
    def ivp_solver(self) -> IVP_Solver[T_co]:
        r"""Initial value problem solver."""
        return cast(IVP_Solver[T_co], solve_ivp)

    @final
    def get_initial_state(self, size: SizeLike = ()) -> T_co:
        r"""Generate (multiple) initial state(s) y₀."""
        # get the initial state
        y0 = self._get_initial_state_impl(size=size)
        # project onto the constraint set
        y0 = self.project_initial_state(y0)
        # validate initial state
        self.validate_initial_state(y0)
        return y0

    @final
    def make_observations(self, sol: Any, /) -> T_co:
        r"""Create observations from the solution."""
        # get observations (add noise))
        obs = self._make_observations_impl(sol)
        # project onto the constraint set
        obs = self.project_observations(obs)
        # validate observations
        self.validate_observations(obs)
        return obs

    @final
    def solve_ivp(self, t: ArrayLike, /, *, y0: ArrayLike) -> T_co:
        r"""Solve the initial value problem."""
        # solve the initial value problem
        sol = self._solve_ivp_impl(t, y0=y0)
        # project onto the constraint set
        sol = self.project_solution(sol)
        # validate solution
        self.validate_solution(sol)
        return sol

    # region implementation ------------------------------------------------------------
    @abstractmethod
    def _get_initial_state_impl(self, *, size: SizeLike = ()) -> T_co:
        r"""Generate (multiple) initial state(s) y₀."""
        ...

    @abstractmethod
    def _make_observations_impl(self, sol: Any, /) -> T_co:
        r"""Create observations from the solution."""
        ...

    def _solve_ivp_impl(self, t: ArrayLike, /, *, y0: ArrayLike) -> T_co:
        r"""Solve the initial value problem."""
        if self.ivp_solver is NotImplemented or self.system is NotImplemented:
            raise NotImplementedError
        if self.ivp_solver is scipy_solve_ivp:
            raise ValueError(
                "scipy.integrate.solve_ivp does not match the IVP_solver Protocol,"
                " since it requires separate bounds [t0,tf] and evaluation points"
                " t_eval. Please use the wrapped version"
                " tsdm.random.generators.solve_ivp instead."
            )
        # solve the initial value problem
        return self.ivp_solver(self.system, t, y0=y0)

    # endregion implementation ---------------------------------------------------------

    # region validation and projection -------------------------------------------------
    # NOTE: These are optional and can be overwritten by subclasses to enforce/validate
    #       constraints on the initial state, solution and observations.

    @property
    def project_initial_state(self) -> SelfMap[T_co]:
        r"""Project the initial state onto the constraint set."""
        return lambda y0: y0

    @property
    def project_observations(self) -> SelfMap[T_co]:
        r"""Project the observations onto the constraint set."""
        return lambda obs: obs

    @property
    def project_solution(self) -> SelfMap:
        r"""Project the solution onto the constraint set."""
        return lambda sol: sol

    @property
    def validate_initial_state(self) -> NullMap[T_co]:
        r"""Validate constraints on the initial state."""
        return lambda y0: None

    @property
    def validate_observations(self) -> NullMap[T_co]:
        r"""Validate constraints on the parameters."""
        return lambda obs: None

    @property
    def validate_solution(self) -> NullMap:
        r"""Validate constraints on the parameters."""
        return lambda sol: None

    # endregion validation and projection ----------------------------------------------


if TYPE_CHECKING:
    scipy_dist: type[Distribution] = rv_continuous
    scipy_solver: IVP_Solver = scipy_solve_ivp
    _solve_ivp: IVP_Solver = solve_ivp