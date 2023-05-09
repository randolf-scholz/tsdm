#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Test of the possibility of inputting a nn.Module as input into a custom autograd function

# %%
import warnings
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import torch
from torch import Tensor, dot, nn
from torch.autograd import grad

# %% [markdown]
# # Setup - Implement a Linear Solver (CGS - Conjugate Gradients Squared)


# %%
class CGS_STATE(NamedTuple):
    """State of the conjugate gradient squared solver."""

    L: Callable[[Tensor], Tensor]
    """The linear function."""
    x: Tensor
    """Vector: Current iterate."""
    r: Tensor
    """Vector: Residual vector."""
    p: Tensor
    """Vector: Search direction."""
    q: Tensor
    """Vector: """
    r0star: Tensor
    """Vector: Initial dual residual vector."""
    rho: Tensor
    """Scalar: Inner Product between r and r0star."""


def cgs_step(state: CGS_STATE) -> CGS_STATE:
    """Perform a single step of the conjugate gradient squared method."""
    # unpack state
    L = state.L
    x = state.x
    r = state.r
    p = state.p
    q = state.q
    r0star = state.r0star
    rho_old = state.rho

    # perform iteration
    rho = dot(r, r0star)
    beta = rho / rho_old
    u = r + beta * q
    p = u + beta * (q + beta * p)
    v = L(p)
    sigma = dot(v, r0star)
    alpha = rho / sigma
    q = u - alpha * v
    r = r - alpha * L(u + q)
    x = x + alpha * (u + q)
    return CGS_STATE(L=L, x=x, r=r, p=p, q=q, r0star=r0star, rho=rho)


def cgs(
    L: nn.Module,
    y: Tensor,
    x0: Optional[Tensor] = None,
    r0star: Optional[Tensor] = None,
    maxiter: int = 100,
    atol: float = 10**-8,
    rtol: float = 10**-5,
) -> CGS_STATE:
    """Solves linear equation L(x)=y."""
    tol = max(atol, rtol * y.norm())
    x0 = torch.zeros_like(y) if x0 is None else x0
    r0 = y - L(x0)
    r0star = r0.clone() if r0star is None else r0star
    p0 = torch.zeros_like(r0)
    q0 = torch.zeros_like(r0)
    rho0 = 1.0  # dot(r0, r0star)
    state = CGS_STATE(L=L, x=x0, r=r0, p=p0, q=q0, r0star=r0star, rho=rho0)

    for it in range(maxiter):
        state = cgs_step(state)

        if state.r.norm() <= tol:
            print(f"Converged after {it} iterations.")
            break
    else:
        warnings.warn(f"No convergence after {maxiter} iterations.")

    residual = (y - L(state.x)).norm().item()
    print(f"Final {residual=:.4}  (r={state.r.norm().item():.4})")
    return state


# %% [markdown]
# ## Test whether CGS works

# %%
N = 8
L = nn.Linear(N, N, bias=False)
# L.weight = nn.Parameter(torch.eye(N) + torch.randn(N, N) / np.sqrt(N))
y = torch.randn(N)
x0 = torch.zeros_like(y)
x_cgs = cgs(L, y).x

# %% [markdown]
# ## Verify against scipy

# %%
from scipy.sparse.linalg import cgs as cgs_scipy

A = L.weight.detach().numpy()
b = y.numpy()
x_ref, r = cgs_scipy(A, b, x0=np.zeros_like(b))
print(f"Final residual: {np.linalg.norm(A @ x_ref - b)}")

diff = np.mean((x_cgs.detach().numpy() - x_ref) ** 2)
print(f"MSE between custom and reference solution: {diff}")


# %% [markdown]
# # Test on a model
#
# We compute gradients for $‖\text{deq-layer}(x)‖^2$.


# %%
def fixed_point_iteration(
    f: nn.Module,
    x: Tensor,
    maxiter: int = 100,
    atol: float = 10**-8,
    rtol: float = 10**-5,
) -> Tensor:
    """Solves $z⁎=f(x，z⁎)$ via FP iteration."""
    z = torch.zeros(f.hidden_size)
    for it in range(maxiter):
        z_new = f(x, z)
        converged = (z_new - z).norm() <= rtol * z.norm() + atol
        z = z_new
        if converged:
            print(f"Converged after {it} iterations.")
            break
    else:
        warnings.warn(f"No convergence after {maxiter} iterations.")
    return z


input_size, hidden_size = 4, 3
model = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
x = torch.randn(input_size)

# %% [markdown]
# ## 1. Automatic differentiation (through the fixed-point iteration)

# %%
model.zero_grad(set_to_none=True)

# forward: fixed point iteration
z = fixed_point_iteration(model, x)

# backward
z.norm().pow(2).backward()
reference_gradients = [w.grad for w in model.parameters()]
print(reference_gradients)

# %% [markdown]
# ## 2. Manual computation

# %%
model.zero_grad(set_to_none=True)

# forward: fixed point iteration
with torch.no_grad():
    z = fixed_point_iteration(model, x)

# backward setup.
outer_grad = 2 * z  # ∂‖z‖²/∂z = 2z
zstar = z.requires_grad_()  # must enable grad to compute ∂f/∂z⁎
fstar = model(x, zstar)

# backward step 1: solve for $g ≔ \Bigl(𝕀 - \frac{∂f}{∂z⁎}\Bigr)^{-⊤} y$
L = lambda g: g - grad(fstar, zstar, g, retain_graph=True)[0]
gstar = cgs(L, outer_grad).x

# compute the outer grad
manual_gradients = grad(fstar, model.parameters(), gstar)

print("MSE between automatic gradients to manual gradients:")
for g1, g2 in zip(reference_gradients, manual_gradients):
    print((g1 - g2).pow(2).mean())


# %% [markdown]
# ## 3. Using `register_hook` (https://implicit-layers-tutorial.org/deep_equilibrium_models/)


# %%
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver):
        super().__init__()
        self.f = f
        self.solver = solver

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z = fixed_point_iteration(self.f, x)
        z = self.f(x, z)

        # set up Jacobian vector product (without additional forward calls)
        zstar = z.clone().detach().requires_grad_()
        fstar = self.f(x, zstar)

        def backward_hook(outer_grad):
            L = lambda g: g - grad(fstar, zstar, g, retain_graph=True)[0]
            gstar = self.solver(L, outer_grad).x
            return gstar

        z.register_hook(backward_hook)
        return z


# %%
DEQ = DEQFixedPoint(model, cgs)
DEQ.zero_grad(set_to_none=True)

print([w.grad for w in DEQ.parameters()])

# forward
y = DEQ(x)

# backward
y.norm().pow(2).backward()
gradients_deq = [w.grad for w in DEQ.parameters()]

print("MSE between automatic gradients to manual gradients:")
for g1, g2 in zip(reference_gradients, gradients_deq):
    print((g1 - g2).pow(2).mean())


# %% [markdown]
# ## 4. Using a custom autograd function (doesn't work)


# %%
class DEQ_Layer(torch.autograd.Function):
    @staticmethod
    def forward(f: nn.Module, x: Tensor, **kwargs: Any) -> Tensor:
        """Compute the fixed point $z⁎ = f(x, z⁎)$."""
        zstar = fixed_point_iteration(f, x, **kwargs)
        return zstar.requires_grad_()

    @staticmethod
    def setup_context(ctx, inputs, output, **_):
        f, x = inputs
        zstar = output
        ctx.save_for_backward(f, x, zstar)

    @staticmethod
    def backward(self, ctx, grad_output):
        f, x, zstar = ctx.saved_tensors

        with torch.enable_grad():
            fstar = f(x, zstar)

        # solve the linear system $(𝕀 - ∂f(x，z⁎)/∂z⁎)ᵀg = y$
        L = lambda g: g - grad(fstar, zstar, g, retain_graph=True)[0]
        gstar = cgs(L, grad_output).x

        # compute the outer grads
        grad_f = [
            (grad(fstar, w, gstar) if w.requires_grad else None) for w in f.parameters()
        ]
        grad_x = grad(fstar, x, gstar) if x.requires_grad else None
        return grad_f, grad_x


# %%
model.zero_grad(set_to_none=True)

# forward
y = DEQ_Layer.apply(model, x).norm().pow(2)

# backward
y.backward()
print([w.grad for w in model.parameters()])  # ✘ no gradients...
model.zero_grad()


# %% [markdown]
# ## 5. Proposed Solution via 'nn.Module.backward'


# %%
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver):
        super().__init__()
        self.f = f
        self.solver = solver

    def forward(self, x: Tensor) -> Tensor:
        return fixed_point_iteration(self.f, x)

    def backward(self, outer_grad):
        """Computes gradients for inputs of forward.

        Since this a nn.Module, we ought to compute gradients both for 'self' and 'x'
        """
        # let's just assume these are automatically captured somehow
        # alternatively, one could make use of buffers.
        x = self.forward.inputs
        z = self.forward.outputs

        # backward step 1: solve for $g ≔ \Bigl(𝕀 - \frac{∂f}{∂z⁎}\Bigr)^{-⊤} y$
        zstar = z.requires_grad_()
        fstar = self.f(x, zstar)
        L = lambda g: g - grad(fstar, zstar, g, retain_graph=True)[0]
        gstar = self.solver(L, outer_grad)

        # compute the outer grads
        # grad_self = grad(fstar, self, gstar)  # <- would be nice to be able to do this.
        grad_self = [
            (grad(fstar, w, gstar) if w.requires_grad else None)
            for w in self.parameters()
        ]
        grad_x = grad(fstar, x, gstar) if x.requires_grad else None
        return grad_self, grad_x


# %% [markdown]
# ## Background: DEQ models
#
# In a Deep Equilibrium model, given input $x$, we use out model $f$, parameterized by $θ$ to compute the fixed point
#
#
# $$ z⁎ = f(z⁎，x，θ) $$
#
# Now, we need the gradients $\frac{∂z⁎}{∂θ}$. Computing the derivative on both sides yields
#
# $$ \frac{∂z⁎}{∂θ}
# = \frac{∂f}{∂z⁎}\frac{∂z⁎}{∂θ} + \frac{∂f}{∂x}\frac{∂x}{∂θ} + \frac{∂f}{∂θ}\frac{∂θ}{∂θ}
# = \frac{∂f}{∂z⁎}\frac{∂z⁎}{∂θ} + \frac{∂f}{∂θ}
# ⟹ \Bigl(𝕀 - \frac{∂f}{∂z⁎}\Bigr)\frac{∂z⁎}{∂θ} = \frac{∂f}{∂θ}
# $$
#
# In particular, the VJP given outer gradient $g$ is
#
# $$ \frac{∂z⁎}{∂θ}^⊤ y = \frac{∂f}{∂θ}^⊤ \Bigl(𝕀 - \frac{∂f}{∂z⁎}\Bigr)^{-⊤} y $$
#
# so, as an intermediate we need to compute
#
# $$ g ≔ \Bigl(𝕀 - \frac{∂f}{∂z⁎}\Bigr)^{-⊤} y ⟺ \Bigl(𝕀 - \frac{∂f}{∂z⁎}\Bigr)^⊤ g = y ⟺ g + \text{VJP}(f, z⁎, g) = y$$
#
# Once we have $g$ we can compute
#
# $$\text{VJP(z⁎，θ，y)} = \text{VJP}(f，θ，g)$$
#
# In summary, the steps for computing the gradients are:
#
# 1. **Forward:** given input $x$, return solution  $z⁎$  of the Fixed Point equation $z = f(z，x，θ)$.
# 2. **Backward:** Given outer gradients y, we need  to compute the gradients $\frac{∂z⁎}{∂θ}$.
#    1. Compute solution $g⁎$ of the linear system $g+\text{VJP}(f, z⁎，g) = y$.
#    2. Compute $\text{VJP(z⁎，θ，y)} = \text{VJP}(f，θ，g⁎)$.In summary, the steps for computing the gradients are:
