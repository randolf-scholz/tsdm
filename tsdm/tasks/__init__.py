r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a split to cater to the question of
forecasting horizons.
"""


from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = []
