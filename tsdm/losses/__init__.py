r"""Implementation of loss functions.

contains object oriented loss functions.
See `tsdm.losses.functional` for functional implementations.
"""

import logging
from typing import Final, Type

from torch import nn

from tsdm.losses import functional
from tsdm.losses._module import ND, NRMSE

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["functional", "LOSSES", "ND", "NRMSE"]

LOSSES: Final[dict[str, Type[nn.Module]]] = {
    "ND": ND,
    "NRMSE": NRMSE,
}
