r"""Test JIT compilation of all metrics."""

import pytest
import torch

from tsdm.metrics import FUNCTIONAL_LOSSES, MODULAR_LOSSES, WeightedMetric
from tsdm.metrics.timeseries import WeightedTimeSeriesLoss


@pytest.mark.parametrize("loss_name", FUNCTIONAL_LOSSES | MODULAR_LOSSES)
def test_metric_jit_scriptable(loss_name: str) -> None:
    r"""Test that all functional metrics can be JIT compiled."""
    NDIM = 5
    NBATCH = 10

    if loss_name in MODULAR_LOSSES:
        loss_class = MODULAR_LOSSES[loss_name]
        if issubclass(loss_class, WeightedMetric | WeightedTimeSeriesLoss):
            weight = torch.rand(NDIM)
            loss_fn = loss_class(weight)
        else:
            loss_fn = loss_class()
        scripted_loss_fn = torch.jit.script(loss_fn)
    else:
        loss_fn = FUNCTIONAL_LOSSES[loss_name]
        scripted_loss_fn = torch.jit.script(loss_fn)

    # Test that the scripted function produces the same output
    x = torch.randn(NBATCH, NDIM)
    xhat = torch.randn(NBATCH, NDIM)
    original_output = loss_fn(x, xhat)
    scripted_output = scripted_loss_fn(x, xhat)
    assert torch.allclose(original_output, scripted_output)


@pytest.mark.parametrize("loss_name", FUNCTIONAL_LOSSES)
def test_metric_compilable(loss_name: str) -> None:
    r"""Test that all functional metrics can be compiled with torch.compile."""
    loss_fn = FUNCTIONAL_LOSSES[loss_name]
    compiled_loss_fn = torch.compile(loss_fn)

    # Test that the scripted function produces the same output
    x = torch.randn(10, 5)
    xhat = torch.randn(10, 5)
    original_output = loss_fn(x, xhat)
    scripted_output = compiled_loss_fn(x, xhat)
    assert torch.allclose(original_output, scripted_output)
