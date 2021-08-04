r"""Functional implementations of loss functions."""

import logging

from tsdm.losses.functional._functional import nd, nrmse


logger = logging.getLogger(__name__)
__all__ = ["LOSSES", "nd", "nrmse"]

LOSSES = {
    "nd": nd,
    "nrmse": nrmse,
}
