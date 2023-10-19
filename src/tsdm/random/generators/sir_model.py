"""Implementation of the SIR model from epidemiology."""

__all__ = ["SIR"]

from dataclasses import KW_ONLY, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsdm.random.generators._generators import IVP_GeneratorBase
from tsdm.random.stats.distributions import Dirichlet
from tsdm.types.aliases import SizeLike


@dataclass
class SIR(IVP_GeneratorBase[NDArray]):
    r"""SIR model from epidemiology.

    .. math::
        \dot{S} &= -β⋅I⋅S       \\
        \dot{I} &= β⋅I⋅S - α⋅I  \\
        \dot{R} &= α⋅I          \\
        S + I + R &= 1

    S: susceptible population (fraction)
    I: infected population (fraction)
    R: recovered population (fraction)
    α: recovery rate (probability of recovery per unit time)
    β: transmission rate (probability of transmission per contact per unit time)
    𝓡₀ = β / α is the basic reproduction number.
    -dS/dt: incidence rate
    """

    _: KW_ONLY
    alpha: float = 0.1
    """Recovery rate."""
    beta: float = 0.5
    """Transmission rate."""

    def _get_initial_state_impl(
        self, size: SizeLike = (), *, weights: ArrayLike = (100, 1, 0)
    ) -> NDArray:
        """Generate (multiple) initial state(s) y₀.

        We simply use a Dirichlet distribution with parameters 100:1:0.
        """
        return Dirichlet.rvs(weights, size=size)

    def _make_observations_impl(
        self, y: NDArray, /, *, noise: float = 0.001
    ) -> NDArray:
        r"""Create observations from the solution.

        We sample from a dirichlet distribution with parameters
        [S, I, R]/noise.
        """
        return Dirichlet.rvs(y / noise)

    def system(self, t: ArrayLike, state: ArrayLike) -> NDArray:
        r"""Vector field of the SIR model.

        .. Signature:: ``[(...B, N), (...B, N, 3) -> (...B, N, 3)``

        sub-signatures:
            - ``[(...,), (2, ) -> (..., 2)``
            - ``[(,), (..., 2) -> (..., 2)``
        """
        t = np.asarray(t)
        state = np.asarray(state)

        S, I, R = np.moveaxis(state, -1, 0)  # noqa: E741
        x = np.stack(
            [
                -self.beta * I * S,
                self.beta * I * S - self.alpha * I,
                self.alpha * I,
            ]
        )
        return np.einsum("..., ...d -> ...d", np.ones_like(t), x)

    def project_solution(self, x: NDArray, /, *, tol: float = 1e-3) -> NDArray:
        r"""Project the solution onto the constraint set."""
        assert x.min() > -tol, "Integrator produced vastly negative values."
        assert x.max() < 1 + tol, "Integrator produced vastly positive values."
        assert np.allclose(x.sum(axis=-1), 1, atol=tol), "Constraints vioalted."

        x = x.clip(0, 1)
        x /= x.sum(axis=-1, keepdims=True)
        return x

    def validate_solution(self, x: NDArray, /) -> None:
        assert (
            x.min() >= 0 and x.max() <= +1 and np.allclose(x.sum(axis=-1), 1)
        ), "Integrator produced invalid values."
