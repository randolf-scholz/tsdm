#!/usr/bin/env python
import torch
from torch import Tensor, jit, nn

device = torch.device("cuda")


class Foo(nn.Module):
    scalar: Tensor

    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
        self.layer = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.scalar * self.layer(x)


model = jit.script(Foo()).to(device=device)

optimizer_config = {
    "lr": 0.001,
    "betas": torch.tensor([0.9000, 0.9990]),
    # "betas": [0.9000, 0.9990],
    "weight_decay": 0.001,
    "eps": 1e-06,
}

optim = torch.optim.Adam(model.parameters(), **optimizer_config)


x = torch.randn(5, 4, device=device)
r = model(x).norm()
r.backward()
optim.step()
