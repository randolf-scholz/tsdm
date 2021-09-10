r"""Tasks are Datasets + Problem Formulation.

Should we use OpenML to define tasks?


A task m,ust encapsulate a well-defined problem formulation.

Things a Task may define:

- splits.
  - either pre-defined indices, or a procedure to generate splits.
- target metric.
- evaluation protocol.
- computational budget.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = []
