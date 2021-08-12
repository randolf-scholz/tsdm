r"""Functional implementations of loss functions."""

import logging
from typing import Final

from tsdm.losses.functional._functional import nd, nrmse

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["LOSSES", "nd", "nrmse"]

LOSSES = {
    "nd": nd,
    "nrmse": nrmse,
}
