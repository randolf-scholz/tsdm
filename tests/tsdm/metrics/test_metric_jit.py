r"""Test JIT compilation of all metrics."""

import pytest
import torch

from tsdm.metrics import FUNCTIONAL_LOSSES


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
