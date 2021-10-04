r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = ["USHCN_GRU_ODE_BAYES"]

import logging
from typing import Literal

from tsdm.datasets import Dataset, USHCN_SmallChunkedSporadic
from tsdm.tasks.tasks import BaseTask

LOGGER = logging.getLogger(__name__)


class USHCN_GRU_ODE_BAYES(BaseTask):
    """DOsztring."""

    dataset: Dataset = USHCN_SmallChunkedSporadic
    test_metric = Literal["MSE", "NLL"]
