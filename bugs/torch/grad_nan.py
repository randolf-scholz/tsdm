#!/usr/bin/env python
# +
# # [Incorrect gradients in NaN-ignoring MSE #89543](https://github.com/pytorch/pytorch/issues/89543)
# +
import torch

N = 4
# create some observations with missing values
y = torch.tensor([1, 2, float("nan"), float("nan")])
x = torch.nn.Parameter(torch.randn(N), requires_grad=True)
w = torch.randn(N)
# -

# ## Absolute value works

# variant 1 correctly produces NaN-free gradient
x.grad = None
r = x - y
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
r = w * r.abs()
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✔

# variant 2 incorrectly produces NaN-gradients
x.grad = None
r = x - y
r = w * r.abs()
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✘


# ## squared error does not work

# variant 1 correctly produces NaN-free gradient
x.grad = None
r = x - y
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
r = w * r**2
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✔


# variant 2 incorrectly produces NaN-gradients
x.grad = None
r = x - y
r = w * r**2
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✘


# # with jacfwd

# +
import functorch
from torch import nn, tensor

x = nn.Parameter(tensor([1.1, 1.2, 1.3]))
y = tensor([1, 2, float("nan")])


def f(x):
    r = x - y
    r = r**2
    m = torch.isnan(y)
    r = torch.where(m, 0.0, r)
    return r.sum() / m.sum()


df = functorch.jacfwd(f)
print(df(x))
df = functorch.jacrev(f)
print(df(x))

# +
import functorch
from torch import nn, tensor

x = nn.Parameter(tensor([1.1, 1.2, 1.3]))
y = tensor([1, 2, float("nan")])


def f(x):
    r = x - y
    m = torch.isnan(y)
    r = torch.where(m, 0.0, r)
    r = r**2
    return r.sum() / m.sum()


df = functorch.jacfwd(f)
print(df(x))
df = functorch.jacrev(f)
print(df(x))

# +
import torch

x = torch.nn.Parameter(torch.tensor([1.1, 1.2, 1.3]))
y = torch.tensor([2, 3, float("nan")])
m = torch.tensor([True, True, False])

x.grad = None
r = (x - y) ** 2
r = torch.where(m, r, 0.0)
loss = r.sum() / m.sum()
loss.backward()
print(x.grad)
assert not any(torch.isnan(x.grad))  # ✔

# +
import jax.numpy as np

x = np.array([1.2, 1.3, 1.4])
y = np.array([2, 3, 4])
m = np.array([False, False, True])


def f(x):
    r = x - y
    r = r**2
    # m = np.isnan(y)
    r = np.where(m, 0.0, r)
    return np.mean(r)


jax.grad(f)(x)
# -

# ## Calculation by hand

# - Case 1:  $f(x) = ∑_i ([m \mathbin{?} x-y : 0]_i)^2$
# - Case 2:  $f(x) = ∑_i [m \mathbin{?} (x-y)^2 : 0]_i$

# Gradient in case 1.
#
# Define intermediate quantities: $r = x - y$, $z = [m \mathbin{?} r : 𝟎]$, $y = ∑ z_i^2$
#
#
# $$\begin{aligned}
# \frac{∂ f(x)}{∂x}
#    &= \frac{∂y}{∂z}∘\frac{∂z}{∂r}∘\frac{∂r}{∂x}
# \\ &= [∆z ↦ ⟨2z∣∆z⟩] ∘ [∆r ↦ [m \mathbin{?} ∆r : 𝟎 ]] ∘ [∆x ↦ ∆x]  & g_z &= 2z
# \\ &= [∆r ↦ ⟨2z∣[m \mathbin{?} ∆r : 𝟎 ]⟩] ∘ [∆x ↦ ∆x]
# \\ \text{(dubious step)} &= [∆r ↦ ⟨[m \mathbin{?}  2z : 𝟎]∣ ∆r⟩]  ∘ [∆x ↦ ∆x]
# \\ &= [∆r ↦ ⟨[m \mathbin{?} 2[m \mathbin{?} x -y : 𝟎] : 𝟎]∣ ∆r⟩]  ∘ [∆x ↦ ∆x]
# \\ &= [∆r ↦ ⟨[m \mathbin{?} 2(x -y) : 𝟎]∣ ∆r⟩]  ∘ [∆x ↦ ∆x]   & g_r &= [m \mathbin{?}  2(x-y) : 𝟎]
# \\ &= [∆x ↦ ⟨[m \mathbin{?}  2(x-y) : 𝟎]∣ ∆x⟩]                         & g_x &= [m \mathbin{?}  2(x-y) : 𝟎]
# \end{aligned}$$

# Gradient in case 2.
#
# Define intermediate quantities: $s = (x-y)^{⊙2}$, $z = [m \mathbin{?} s : 𝟎]$, $y = ∑ z_i$
#
#
# $$\begin{aligned}
# \frac{∂ f(x)}{∂x}
#    &= \frac{∂y}{∂z}∘\frac{∂z}{∂s}∘\frac{∂s}{∂x}
# \\ &= [∆z ↦ ⟨𝟏∣∆z⟩] ∘ [∆s ↦ [m \mathbin{?} ∆s : 𝟎 ]] ∘ [∆x ↦ 2(x-y)⊙∆x]  & g_z &= 𝟏
# \\ &= [∆s ↦ ⟨𝟏∣[m \mathbin{?} ∆s : 𝟎 ]⟩] ∘ [∆x ↦  2(x-y)⊙∆x]
# \\ \text{(dubious step)} &= [∆s ↦ ⟨[m \mathbin{?}  𝟏 : 𝟎]∣ ∆s⟩] ∘ [∆x ↦ 2(x-y)⊙∆x]      & g_s &= [m \mathbin{?}  𝟏 : 𝟎]
# \\ &= [∆x ↦ ⟨[m \mathbin{?}  𝟏 : 𝟎]∣ 2(x-y)⊙∆x⟩]
# \\ &= [∆x ↦ ⟨[m \mathbin{?}  𝟏 : 𝟎]⊙2(x-y)∣ ∆x⟩]
# \\\text{(incorrect)} &= [∆x ↦ ⟨[m \mathbin{?}  𝟏⊙2(x-y) : 𝟎⊙2(x-y)]∣ ∆x⟩]           & g_x &= [m \mathbin{?}  2(x-y) : 𝟎⊙2(x-y)]
# \\\text{(correct)} &= [∆x ↦ ⟨[m \mathbin{?}  𝟏⊙2(x-y) : 𝟎]∣ ∆x⟩]           & g_x &= [m \mathbin{?}  2(x-y) : 𝟎]
# \end{aligned}$$

# ## Correct backwad of the `where` operator
#
# I suspect this is due to an error in the definition of the where backward. Thinking about it a bit, I came up with the following formula for the correct backward of the `where` operator:
#
# Let $y=[m \mathbin{?} a : b]$ and $a$ and $b$ functions of $x$. then
#
# $$\begin{aligned}
# \frac{∂y}{∂x} &= \frac{∂y}{∂(a, b)} ∘ \frac{∂(a,b)}{∂x} \\
# &= \Bigl[ (∆a, ∆b) ⟼ [m  \mathbin{?} {\color{green}{[m \mathbin{?} ∆a : 𝟎 ]}} : {\color{green}{[m \mathbin{?} 𝟎 : ∆b ]}}\Bigr]∘[∆x ⟼ \bigl(\tfrac{∂a}{∂x}{∆x}， \tfrac{∂b}{∂x}{∆x}\bigr) ] \\
# \text{(correct)}&= \Bigl[ ∆x  ⟼ [m  \mathbin{?} {\color{green}{[m \mathbin{?} \tfrac{∂a}{∂x}{∆x} : 𝟎 ]}} : {\color{green}{[m \mathbin{?} 𝟎 : \tfrac{∂b}{∂x}{∆x} ]}}\Bigr] \\
# \text{(incorrect)}&= \Bigl[ ∆x  ⟼ [m  \mathbin{?} {\color{red}{\tfrac{∂a}{∂x}{∆x}}} : {\color{red}{\tfrac{∂b}{∂x}{∆x}}}]\Bigr]   \qquad\textbf{this simplification is incorrect if there are NaN values!}
# \end{aligned}$$

# ## Correct vector-Jacobian product for the backward
#
#
# $$\begin{aligned}
# \frac{∂ℓ}{∂y}\frac{∂y}{∂x} &= [∆y ↦ \frac{∂ℓ}{∂y}{∆y}] ∘ \Bigl[ ∆x  ⟼ [m  \mathbin{?} {\color{green}{[m \mathbin{?} \tfrac{∂a}{∂x}{∆x} : 𝟎 ]}} : {\color{green}{[m \mathbin{?} 𝟎 : \tfrac{∂b}{∂x}{∆x} ]}}\Bigr] \\
# &= \Bigl[ (∆a, ∆b) ⟼ [m  \mathbin{?} {\color{green}{[m \mathbin{?} ∆a : 𝟎 ]}} : {\color{green}{[m \mathbin{?} 𝟎 : ∆b ]}}\Bigr]∘[∆x ⟼ \bigl(\tfrac{∂a}{∂x}{∆x}， \tfrac{∂b}{∂x}{∆x}\bigr) ] \\
# \text{(correct)}&= \Bigl[ ∆x  ⟼ [m  \mathbin{?} {\color{green}{[m \mathbin{?} \tfrac{∂a}{∂x}{∆x} : 𝟎 ]}} : {\color{green}{[m \mathbin{?} 𝟎 : \tfrac{∂b}{∂x}{∆x} ]}}\Bigr] \\
# \text{(incorrect)}&= \Bigl[ ∆x  ⟼ [m  \mathbin{?} {\color{red}{\tfrac{∂a}{∂x}{∆x}}} : {\color{red}{\tfrac{∂b}{∂x}{∆x}}}]\Bigr]
# \end{aligned}$$
#
#
