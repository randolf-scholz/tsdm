r"""METRICS (non-differentiable losses).

We define the following

1. A metric is a  function

    .. math::
        𝔪： 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥0}

    such that

    - $𝔪(\{ (y_i, \hat{y}_i) ∣ i=1:n \}) = 0$ if and only if $y_i=\hat{y}_i∀i$

2. A metric is called decomposable, if it can be written as a function

    .. math
        𝔪 = Ψ∘(ℓ×𝗂𝖽)
        ℓ： 𝓨×𝓨 ⟶ ℝ_{≥0}
        Ψ： 𝓟_0(ℝ_{≥0}) ⟶ ℝ_{≥0}

    I.e. the function $ℓ$ is applied element-wise to all pairs $(y, \hat{y}$ and the function $Ψ$
    "accumulates" the results. Oftentimes, $Ψ$ is just the sum/mean/expectation value, although
    other accumulations such as the median value are also possible.

3. A metric is called instance-wise, if it can be written in the form

    .. math::
        𝔪： 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥ 0}, 𝔪(\{(y_i, \hat{y}_i) ∣  i=1:n \})
        = ∑_{i=1}^n ω(i, n)ℓ(y_i, \hat{y}_i)

4. A metric is called a loss-function, if and only if

   - It is differentiable almost everywhere.
   - It is non-constant, at least on some open set.

Note that in the context of time-series, we allow the accumulator to depend on the time variable.
"""

__all__ = [
    # Sub-Module
    "functional",
    "modular",
    # Types
    "FunctionalMetric",
    "ModularMetric",
    "Metric",
    # Constants
    "FunctionalMetrics",
    "ModularMetrics",
    "METRICS",
]

import logging
from typing import Final, Union

from tsdm.metrics import functional, modular
from tsdm.metrics.functional import FunctionalMetric, FunctionalMetrics
from tsdm.metrics.modular import ModularMetric, ModularMetrics

__logger__ = logging.getLogger(__name__)


Metric = Union[FunctionalMetric, ModularMetric]
r"""Type hint for metrics."""

METRICS: Final[dict[str, Union[FunctionalMetric, type[ModularMetric]]]] = {
    **FunctionalMetrics,
    **ModularMetrics,
}
r"""Dictionary of all available  metrics."""
