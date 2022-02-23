#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, jit


class MWE(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(2 * input_size, output_size)

    def forward(self, x):
        r"""[..., input_size] -> [..., output_size]"""
        x = torch.cat([x, x], dim=-1)
        # print("")  # FIXME: Commenting this line causes RunTimeError
        x = self.linear(x)
        return x


xdim = 7
model = jit.script(MWE(xdim, 10))  # bug does not happen without jit

for k in range(100):
    num_observations = torch.randint(0, 3, (1,)).item()
    x = torch.randn(num_observations, xdim)
    print(f"Sample {k=} of shape {x.shape}")
    model.zero_grad()
    z = model(x)
    z.norm().backward()
