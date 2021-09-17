r"""Metric and Losses.

We define the following

1. A metric is a  function

    .. math::
        𝔪\colon 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥ 0}

    such that

    - $𝔪(\{(y_i, \hat{y}_i) ∣  i=1:n \}) = 0$ if and only if $y_i=\hat{y}_i∀i$

2. A metric is called decomposable, if it can be written as a function

    .. math
        𝔪 = Ψ∘(ℓ×𝗂𝖽)
        ℓ\colon 𝓨×𝓨 ⟶ ℝ_{≥ 0}
        Ψ\colon 𝓟_0(ℝ_{≥0}) ⟶ ℝ_{≥0}

    I.e. the function $ℓ$ is applied element-wise to all pairs $(y, \hat{y}$ and the function $Ψ$ 
    "accumulates" the results. Oftentimes, $Ψ$ is just the sum/mean/expectation value, although
    other accumulations such as the median value are also possible.

3. A metric is called instance-wise, if it can be written in the form

    .. math::
        𝔪\colon 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥ 0}, 𝔪(\{(y_i, \hat{y}_i) ∣  i=1:n \}) 
        = ∑_{i=1}^n ω(i, n)ℓ(y_i, \hat{y}_i)

4. A metric is called a loss-function, if and only if

   - It is differentiable almost everywhere.
   - It is non-constant, at least on some open set.
   
   
Note that in the context of time-series, we allow the accumulator to depend on 
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = []
