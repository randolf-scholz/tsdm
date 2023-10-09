#!/usr/bin/env python
import torch
from torch import Tensor, jit, nn

device = torch.device("cuda")
model = nn.Linear(4, 4).to(device=device)

# optimizer_config = {
#     "lr": 0.001,
#     "betas": torch.tensor([0.9000, 0.9990]),
# }

optimizer_config = {
    "lr": 0.001,
    "betas": torch.tensor([0.9000, 0.9990]),
    "foreach": False,
    # "fused": False,
}

optim = torch.optim.Adam(model.parameters(), **optimizer_config)
x = torch.randn(5, 4, device=device)
model(x).norm().backward()
optim.step()
