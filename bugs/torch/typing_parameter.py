from typing import assert_type, reveal_type

import torch

w = torch.randn(3)
p = torch.nn.Parameter(w)
assert_type(p, torch.nn.Parameter)  # ❌ expected "Parameter" but received "Tensor"
reveal_type(torch.nn.Parameter.__new__)
# (self: type[Self@TensorBase], *args: Unknown, **kwargs: Unknown) -> Tensor
