r"""Generators for synthetic time series datasets.

Unlike `tsdm.datasets`, which provides static datasets, this module provides
configurable generators for sampling synthetic data from a ground-truth
distribution $p(x, y)$. Such distributions can also be used to study Bayes error,
the minimum achievable prediction error.

Note:
    Initial conditions select a solution of a differential equation when the
    relevant existence and uniqueness assumptions hold. Replacing initial
    conditions with an optimization constraint requires a separate analysis
    of existence and uniqueness.
"""

__all__ = [
    # Types
    "Size",
    # ABCs & Protocols
    "FrozenIVPSolver",
    "IVP_Generator",
    "IVP_Solver",
    "ODE",
    # Classes
    "BaseIVP_Generator",
    "ScipyIVPSolver",
    # Functions
    "solve_ivp",
]

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, asdict, dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, final, runtime_checkable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp as scipy_solve_ivp

from tsdm.constants import RNG
from tsdm.pprint import pprint_repr
from tsdm.random.distributions import TimeSeriesRV

type Size = int | tuple[int, ...]
r"""Type alias for a size specified as an integer or a tuple of dimensions.

Note:
    `None` is not included in this alias. Sampling APIs may accept `size=None`
    separately to request a scalar.
"""


class ODE[T](Protocol):  # +T
    r"""Represents a system of ordinary differential equations."""

    @abstractmethod
    # Float[$N], Float[..., $N,*D] -> Float[..., $N,*D]
    def __call__(self, /, t: ArrayLike, state: ArrayLike) -> T:
        r"""Evaluate the vector field at the given time and state.

        Supported input and output shapes include:

        - `[(), (..., *D)] -> (..., *D)`
        - `[(...,), (*D,)] -> (..., *D)`
        - `[(), (*D,)] -> (*D,)`

        Args:
            t: Scalar time or array of timestamps.
            state: State of the system at time $t$.

        Returns:
            The vector field $f(t, y(t))$ at time $t$ and state $y(t)$.
        """
        ...


class IVP_Solver[T](Protocol):  # +T
    r"""Protocol for initial value problem solvers.

    Adapters can expose solvers from different libraries through this interface.
    The underlying libraries use different conventions:

    - `scipy.integrate.odeint` accepts a function of `(y, t)` by default,
        or `(t, y)` with `tfirst=True`.
    - `scipy.integrate.solve_ivp` and `torchdiffeq.odeint` accept a function
        of `(t, y)`.
    - `torchsde.sdeint` accepts an SDE object with drift method `f(t, y)`
        and diffusion method `g(t, y)`.
    - `brainpy.sdeint` provides decorated integration functions.

    Note:
        This protocol takes evaluation times as its second positional argument and
        requires `y0` as a keyword argument. Adapters must translate that convention
        to the underlying solver's signature.
    """

    @abstractmethod
    # Float[$N], Float[..., *D] -> Float[..., $N,*D]
    def __call__(self, system: ODE | Any, t: ArrayLike, /, *, y0: ArrayLike) -> T:
        r"""Solve the initial value problem.

        Args:
            system: Object representing the dynamics of the system.
            t: Sorted sequence of $N$ evaluation times.
            y0: Initial state at `t[0]`, tensor of shape `(..., *D)`.

        Returns:
            Array-like object `y[tᵢ]` containing the solution of the initial value problem.
        """
        ...


class FrozenIVPSolver[T](Protocol):
    r"""Protocol for an IVP solver with a fixed system."""

    def __call__(self, t: T, /, *, y0: T) -> T: ...


@pprint_repr
@dataclass(slots=True, frozen=True)
class ScipyIVPSolver(FrozenIVPSolver[NDArray]):
    r"""Wrap `scipy.integrate.solve_ivp` with a fixed system and solver options."""

    class METHODS(StrEnum):
        r"""Methods for solving the initial value problem."""

        RK45 = "RK45"
        RK23 = "RK23"
        DOP853 = "DOP853"
        Radau = "Radau"
        BDF = "BDF"

    type METHOD = Literal["RK45", "RK23", "DOP853", "Radau", "BDF"]

    system: ODE

    _: KW_ONLY

    jac: Callable[[NDArray, NDArray], NDArray] | None = None
    method: METHOD = "RK45"
    dense_output: bool = False
    vectorized: bool = False
    first_step: float | None = None
    max_step: float | None = None
    min_step: float | None = None
    rtol: float = 1e-3
    atol: float = 1e-6

    def __call__(self, t: ArrayLike, /, *, y0: ArrayLike, **kwargs: Any) -> NDArray:
        r"""Solve the initial value problem."""
        options = asdict(self)
        system = options.pop("system")
        options |= kwargs
        return solve_ivp(system, t, y0=y0, **options)


def solve_ivp(system: ODE, t: ArrayLike, /, *, y0: ArrayLike, **kwargs: Any) -> NDArray:
    r"""Wrapped version of `scipy.integrate.solve_ivp` that matches the `IVP_Solver` protocol."""
    y0 = np.asarray(y0)
    t_eval = np.asarray(t)
    t_span = (t_eval.min(), t_eval.max())
    sol = scipy_solve_ivp(system, t_span=t_span, y0=y0, t_eval=t_eval, **kwargs)
    # NOTE: output shape: (d, n_timestamps), move time axis to the front
    return np.moveaxis(sol.y, -1, 0)


@runtime_checkable
class IVP_Generator[T: ArrayLike](TimeSeriesRV[T], Protocol):  # +T
    r"""Protocol for generators that solve initial value problems (IVPs).

    Implementations provide the following methods:

    - `get_initial_state` generates initial states $x₀$.
    - `solve_ivp` solves the initial value problem.
    - `make_observations` creates observations from the solution.

    The framework is intended for ODEs, SDEs, and discrete-time systems. A generator
    may incorporate distributions for the initial state, observation noise, and
    process noise.

    Example:  A nonlinear ODE with an observation function
        $$\begin{aligned}
            \dot{x}(t) &= f(t, x(t)) \\
            y(t) &= g(t, x(t))
        \end{aligned}$$
    """

    # region abstract methods ----------------------------------------------------------
    @abstractmethod
    def get_initial_state(self, size: Size = ()) -> T: ...

    @abstractmethod
    def make_observations(self, sol: Any, /) -> T: ...

    @abstractmethod
    def solve_ivp(self, t: ArrayLike, /, *, y0: ArrayLike) -> Any: ...

    @abstractmethod
    def set_rng(self, random_state: int | Generator | None, /) -> None:
        r"""Set the internal random number generator of the IVP_Generator."""
        ...

    # endregion abstract methods -------------------------------------------------------

    # region mixin methods -------------------------------------------------------------
    def rvs(
        self,
        t: ArrayLike,
        size: Size = (),
        *,
        random_state: int | Generator | None = None,
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


class BaseIVP_Generator(IVP_Generator[NDArray]):
    r"""Base class for IVP generators using NumPy arrays."""

    rng: Generator = RNG
    r"""Internal random number generator."""

    @property
    def ivp_solver(self) -> IVP_Solver[NDArray]:
        r"""Initial value problem solver."""
        return solve_ivp

    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""System of differential equations."""
        raise NotImplementedError

    # region implementation ------------------------------------------------------------
    @abstractmethod
    def _get_initial_state_impl(self, *, size: Size = ()) -> NDArray:
        r"""Generate one or more initial states $y₀$."""
        ...

    @abstractmethod
    def _make_observations_impl(self, state: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        ...

    def _solve_ivp_impl(self, t: ArrayLike, /, *, y0: ArrayLike) -> NDArray:
        r"""Solve the initial value problem."""
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
    def set_rng(self, random_state: int | Generator | None, /) -> None:
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
        r"""Generate one or more initial states $y₀$."""
        # get the initial state
        y0 = self._get_initial_state_impl(size=size)
        # project onto the constraint set
        y0 = self.project_initial_state(y0)
        # validate initial state
        self.validate_initial_state(y0)
        return y0

    @final
    def make_observations(self, state: NDArray, /) -> NDArray:
        r"""Create observations from the solution."""
        # get observations (add noise)
        obs = self._make_observations_impl(state)
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

    def project_solution(self, state: NDArray, /) -> NDArray:
        r"""Project the solution onto the constraint set."""
        return state

    def validate_initial_state(self, y0: NDArray, /) -> None:
        r"""Validate constraints on the initial state."""

    def validate_observations(self, obs: NDArray, /) -> None:
        r"""Validate constraints on the observations."""

    def validate_solution(self, state: NDArray, /) -> None:
        r"""Validate constraints on the solution."""

    # endregion validation and projection ----------------------------------------------
