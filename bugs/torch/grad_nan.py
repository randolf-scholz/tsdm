#!/usr/bin/env python

import torch

N = 4
# create some observations with missing values
y = torch.tensor([1, 2, float("nan"), float("nan")])
x = torch.nn.Parameter(torch.randn(N), requires_grad=True)
w = torch.randn(N)

# variant 1 correctly produces NaN-free gradient
r = x - y
m = torch.isnan(y)
r = torch.where(m, 0.0, r).abs()
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✔

# variant 2 incorrectly produces NaN-gradients
x.grad = None
r = w * (x - y).abs()
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✘


# variant 1 correctly produces NaN-free gradient
r = x - y
m = torch.isnan(y)
r = w * torch.where(m, 0.0, r) ** 2
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✔


# variant 2 incorrectly produces NaN-gradients
x.grad = None
r = x - y
r = w * r * r
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✘


# variant 2 incorrectly produces NaN-gradients
x.grad = None
r = w * (x - y) ** 2
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✘
