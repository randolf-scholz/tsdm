r"""Functional implementations of loss functions."""

import logging
from typing import Final

from tsdm.losses.functional.functional import (
    INSTANCE_WISE,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
)

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = ["LOSSES", "nd", "nrmse", "q_quantile", "q_quantile_loss"]

LOSSES: Final[dict[str, INSTANCE_WISE]] = {
    "nd": nd,
    "nrmse": nrmse,
    "q_quantile": q_quantile,
    "q_quantile_loss": q_quantile_loss,
}
