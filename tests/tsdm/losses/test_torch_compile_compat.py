r"""Test compilation of all metrics."""

import pytest
import torch

from tsdm.losses import FUNCTIONAL_VECTOR_LOSSES


@pytest.mark.parametrize("loss_name", FUNCTIONAL_VECTOR_LOSSES)
def test_loss_compilable(loss_name: str) -> None:
    r"""Test that all functional metrics can be compiled with torch.compile."""
    loss_fn = FUNCTIONAL_VECTOR_LOSSES[loss_name]
    compiled_loss_fn = torch.compile(loss_fn)

    # Test that the compiled function produces the same output
    targets = torch.randn(10, 5)
    predictions = torch.randn(10, 5)
    original_output = loss_fn(predictions=predictions, targets=targets)
    compiled_output = compiled_loss_fn(predictions=predictions, targets=targets)
    assert torch.allclose(original_output, compiled_output)
