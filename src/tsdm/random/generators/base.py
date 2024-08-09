r"""Generators for synthetic time series dataset.

Contrary to `tsdm.dataset`, which contains static, real-world dataset, this module
contains generators for synthetic dataset. By design, each generator consists of

- Configuration parameters, e.g. number of dimensions etc.
- Allows sampling from the data ground truth distribution $p(x,y)$.
- Allows estimating the Bayes Error, i.e. the best performance possible on the dataset.
"""

__all__ = [
    # ABCs & Protocols
    "FrozenIVPSolver",
    "IVP_Generator",
    "IVP_Solver",
    "ODE",
    # Classes
    "ScipyIVPSolver",
    "IVP_GeneratorBase",
    # Functions
    "solve_ivp",
]

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, asdict, dataclass
from typing import Any, Literal, Optional, Protocol, final, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp as scipy_solver

from tsdm.constants import RNG
from tsdm.random.distributions import TimeSeriesRV
from tsdm.types.aliases import Size
from tsdm.utils.decorators import pprint_repr


@runtime_checkable
class ODE[T](Protocol):  # +T
    r"""Represents a system of ordinary differential equations."""

    @abstractmethod
    def __call__(self, /, t: ArrayLike, state: ArrayLike) -> T:
        r"""Evaluate the vector field at given time and state.

        .. signature:: ``[(N,), (..., N, *D) -> (..., N, *D)``

        Sub-signatures:
            - ``[(,), (..., *D) -> (..., *D)``
            - ``[(...,), (*D,) -> (..., *D)``
            - ``[(,), (*D,) -> (*D,)``

        Args:
            t: list of time stamps
            state: state of the system at time t

        Returns:
            $f(t, y(t))$ value of the vector field at time $t$ and state $y(t)$.
        """
        ...


@runtime_checkable
class IVP_Solver[T](Protocol):  # +T
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
    def __call__(self, system: ODE | Any, t: ArrayLike, /, *, y0: ArrayLike) -> T:
        r"""Solve the initial value problem.

        .. signature:: ``[(N,), (..., *D)] -> (..., N, *D)``

        Args:
            system: Some object that represents the dynamics of the systems.
            t: sorted list of N time points at which to solve for y.
            y0: Initial state at t[0], tensor of shape `(..., *D)`.

        Returns:
            array-like object y[t_i] containing the solution of the initial value problem.
        """
        ...


class FrozenIVPSolver[T](Protocol):
    r"""Frozen version of the IVP_Solver Protocol."""

    def __call__(self, t: T, /, *, y0: T) -> T:
        r"""Solve the initial value problem."""
        ...


@pprint_repr
@dataclass(frozen=True)
class ScipyIVPSolver(FrozenIVPSolver[NDArray]):
    r"""Wrapped version of `scipy.integrate.solve_ivp` that matches the IVP_solver Protocol."""

    system: ODE

    _: KW_ONLY

    jac: Optional[Callable[[NDArray, NDArray], NDArray]] = None
    method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF"] | str = "RK45"
    dense_output: bool = False
    vectorized: bool = False
    first_step: Optional[float] = None
    max_step: Optional[float] = None
    min_step: Optional[float] = None
    rtol: float = 1e-3
    atol: float = 1e-6

    def __call__(self, t: ArrayLike, /, *, y0: ArrayLike, **kwargs: Any) -> NDArray:
        r"""Solve the initial value problem."""
        t_eval = np.asarray(t)
        t_span = (t_eval.min(), t_eval.max())
        options = asdict(self)
        system = options.pop("system")
        options |= kwargs
        sol = scipy_solver(system, t_span=t_span, y0=y0, t_eval=t_eval, **options)
        # NOTE: output shape: (d, n_timestamps), move time axis to the front
        return np.moveaxis(sol.y, -1, 0)


def solve_ivp(system: ODE, t: ArrayLike, /, *, y0: ArrayLike, **kwargs: Any) -> NDArray:
    r"""Wrapped version of `scipy.integrate.solve_ivp` that matches the IVP_solver Protocol."""
    t_eval = np.asarray(t)
    t_span = (t_eval.min(), t_eval.max())
    sol = scipy_solver(system, t_span=t_span, y0=y0, t_eval=t_eval, **kwargs)
    # NOTE: output shape: (d, n_timestamps), move time axis to the front
    return np.moveaxis(sol.y, -1, 0)


@runtime_checkable
class IVP_Generator[T: ArrayLike](TimeSeriesRV[T], Protocol):  # +T
    r"""Protocol for Generators that solve Initial Value Problems (IVP).

    Needs to implement the following things:

    - `get_initial_state` to generate initial states x₀
    - `solve_ivp` to solve the initial value problem
    - `make_observations` to create observations from the solution

    The goal is a general framework that works for both ODEs, SDEs, and discrete-time systems.
    Thus, an IVP-Generator may incorporate multiple random distributions:

    - a distribution for the initial state x₀
    - a distribution for the observation noise
    - a distribution for the process noise

    Examples:
        Nonlinear ODE

        .. math::
             \dot{x}(t) &= f(t, x(t)) \\
                   y(t) &= g(t, x(t))
    """

    # region abstract methods ----------------------------------------------------------
    @abstractmethod
    def get_initial_state(self, size: Size = ()) -> T:
        r"""Generate (multiple) initial state(s) y₀."""
        ...

    @abstractmethod
    def make_observations(self, sol: Any, /) -> T:
        r"""Create observations from the solution."""
        ...

    @abstractmethod
    def solve_ivp(self, t: ArrayLike, /, *, y0: ArrayLike) -> Any:
        r"""Solve the initial value problem."""
        ...

    @abstractmethod
    def set_rng(self, random_state: Optional[int | Generator], /) -> None:
        r"""Set the internal random number generator of the IVP_Generator."""
        ...

    # endregion abstract methods -------------------------------------------------------

    # region mixin methods -------------------------------------------------------------
    def rvs(
        self,
        t: ArrayLike,
        size: Size = (),
        *,
        random_state: Optional[int | Generator] = None,
    ) -> T:
        r"""Random variates of the given type."""
        # set the random state
        self.set_rng(random_state)

        # get the initial state
        y0 = self.get_initial_state(size=size)

        # solve the initial value problem
        sol = self.solve_ivp(t, y0=y0)

        # get observations (add noise)
        obs = self.make_observations(sol)

        return obs

    # endregion mixin methods ----------------------------------------------------------


class IVP_GeneratorBase(IVP_Generator[NDArray]):
    r"""Base class for IVP_Generators based on numpy."""

    rng: Generator = RNG
    r"""the internal Random Number Generator."""

    @property
    def ivp_solver(self) -> IVP_Solver[NDArray]:
        r"""Initial value problem solver."""
        return solve_ivp

    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""System of differential equations."""
        raise NotImplementedError

    system = NotImplemented  # noqa: F811

    # region implementation ------------------------------------------------------------
    @abstractmethod
    def _get_initial_state_impl(self, *, size: Size = ()) -> NDArray:
        r"""Generate (multiple) initial state(s) y₀."""
        ...

    @abstractmethod
    def _make_observations_impl(self, sol: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        ...

    def _solve_ivp_impl(self, t: ArrayLike, /, *, y0: ArrayLike) -> NDArray:
        r"""Solve the initial value problem."""
        if self.ivp_solver is NotImplemented or self.system is NotImplemented:
            raise NotImplementedError
        if self.ivp_solver is scipy_solver:
            raise ValueError(
                "scipy.integrate.solve_ivp does not match the IVP_solver Protocol,"
                " since it requires separate bounds [t0,tf] and evaluation points"
                " t_eval. Please use the wrapped version"
                " tsdm.random.generators.solve_ivp instead."
            )
        # solve the initial value problem
        return self.ivp_solver(self.system, t, y0=y0)

    # endregion implementation ---------------------------------------------------------
    def set_rng(self, random_state: Optional[int | Generator], /) -> None:
        r"""Set the internal random number generator of the IVP_Generator."""
        match random_state:
            case (None | int()) as seed:
                self.rng = np.random.default_rng(seed)
            case Generator() as generator:
                self.rng = generator
            case _:
                raise TypeError(f"Unsupported {type(random_state)=}")

    @final
    def get_initial_state(self, size: Size = ()) -> NDArray:
        r"""Generate (multiple) initial state(s) y₀."""
        # get the initial state
        y0 = self._get_initial_state_impl(size=size)
        # project onto the constraint set
        y0 = self.project_initial_state(y0)
        # validate initial state
        self.validate_initial_state(y0)
        return y0

    @final
    def make_observations(self, sol: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        # get observations (add noise)
        obs = self._make_observations_impl(sol)
        # project onto the constraint set
        obs = self.project_observations(obs)
        # validate observations
        self.validate_observations(obs)
        return obs

    @final
    def solve_ivp(self, t: ArrayLike, /, *, y0: ArrayLike) -> NDArray:
        r"""Solve the initial value problem."""
        # solve the initial value problem
        sol = self._solve_ivp_impl(t, y0=y0)
        # project onto the constraint set
        sol = self.project_solution(sol)
        # validate solution
        self.validate_solution(sol)
        return sol

    # region validation and projection -------------------------------------------------
    # NOTE: These are optional and can be overwritten by subclasses to enforce/validate
    #       constraints on the initial state, solution and observations.

    def project_initial_state(self, y0: NDArray, /) -> NDArray:
        r"""Project the initial state onto the constraint set."""
        return y0

    def project_observations(self, obs: NDArray, /) -> NDArray:
        r"""Project the observations onto the constraint set."""
        return obs

    def project_solution(self, sol: NDArray, /) -> NDArray:
        r"""Project the solution onto the constraint set."""
        return sol

    def validate_initial_state(self, y0: NDArray, /) -> None:
        r"""Validate constraints on the initial state."""

    def validate_observations(self, obs: NDArray, /) -> None:
        r"""Validate constraints on the parameters."""

    def validate_solution(self, sol: NDArray, /) -> None:
        r"""Validate constraints on the parameters."""

    # endregion validation and projection ----------------------------------------------
