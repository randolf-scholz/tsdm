#!/usr/bin/env python
"""Test whether the JIT compiler optimizes out if/else with Final."""

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, jit, nn


def test_jit_optimization() -> None:
    """Checks that if/else with Final is optimized out by JIT."""

    class Foo(nn.Module):
        use_relu: Final[bool]

        def __init__(self, use_relu: bool) -> None:
            super().__init__()
            self.use_relu = use_relu

        def forward(self, x: Tensor) -> Tensor:
            if self.use_relu:
                return F.relu(x)
            return torch.tanh(x)

    model = Foo(use_relu=False)
    scripted = jit.script(model)
    print(scripted.code)
    assert "relu" not in scripted.code
    assert "tanh" in scripted.code

    model = Foo(use_relu=True)
    scripted = jit.script(model)
    print(scripted.code)
    assert "relu" in scripted.code
    assert "tanh" not in scripted.code


def _main() -> None:
    test_jit_optimization()


if __name__ == "__main__":
    _main()
