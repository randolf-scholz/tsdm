"""Test whether the JIT compiler optimizes out if/else with Final."""

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, jit, nn


def test_jit_optimization() -> None:
    r"""Checks that if/else with Final is optimized out by JIT."""

    class Foo(nn.Module):
        r"""A simple module with a conditional."""

        use_relu: Final[bool]

        def __init__(self, *, use_relu: bool) -> None:
            super().__init__()
            self.use_relu = use_relu

        def forward(self, x: Tensor) -> Tensor:
            r"""Forward pass."""
            if self.use_relu:
                return F.relu(x)
            return torch.tanh(x)

    # check that compiled model does not contain if-clause
    model_with_tanh = Foo(use_relu=False)
    scripted_with_tanh = jit.script(model_with_tanh)
    tanh_code: str = scripted_with_tanh.code  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
    print(f"\n{'-' * 80}\n{'model without relu'}\n{tanh_code}\n{'-' * 80}")
    assert "relu" not in tanh_code
    assert "tanh" in tanh_code

    # check that compiled model does not contain if-clause
    model_with_relu = Foo(use_relu=True)
    scripted_with_relu = jit.script(model_with_relu)
    relu_code: str = scripted_with_relu.code  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
    print(f"\n{'-' * 80}\n{'model with relu'}\n{relu_code}\n{'-' * 80}")
    assert "relu" in relu_code
    assert "tanh" not in relu_code

    # check remaining properties
    for prop in [
        "code",
        "code_with_constants",
        "graph",
        "inlined_graph",
        "original_name",
    ]:
        assert hasattr(scripted_with_relu, prop)
        attr = getattr(scripted_with_relu, prop)
        print(f"\nscripted.{prop}<{type(attr)}> = {attr!r}")
