"""Test whether the JIT compiler optimizes out if/else with Final."""

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, jit, nn


def test_jit_optimization() -> None:
    """Checks that if/else with Final is optimized out by JIT."""

    class Foo(nn.Module):
        """A simple module with a conditional."""

        use_relu: Final[bool]

        def __init__(self, use_relu: bool) -> None:
            super().__init__()
            self.use_relu = use_relu

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass."""
            if self.use_relu:
                return F.relu(x)
            return torch.tanh(x)

    model = Foo(use_relu=False)
    scripted = jit.script(model)

    for prop in [
        "code",
        "code_with_constants",
        "graph",
        "inlined_graph",
        "original_name",
    ]:
        attr = getattr(scripted, prop)
        print(f"\nscripted.{prop}<{type(attr)}> = {attr!r}")
    assert "relu" not in scripted.code  # pyright: ignore[reportGeneralTypeIssues]
    assert "tanh" in scripted.code  # pyright: ignore[reportGeneralTypeIssues]

    model = Foo(use_relu=True)
    scripted = jit.script(model)
    print(scripted.code)  # pyright: ignore[reportGeneralTypeIssues]
    assert "relu" in scripted.code  # pyright: ignore[reportGeneralTypeIssues]
    assert "tanh" not in scripted.code  # pyright: ignore[reportGeneralTypeIssues]
