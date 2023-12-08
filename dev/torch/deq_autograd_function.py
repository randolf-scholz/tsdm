#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Test of the possibility of inputting a nn.Module as input into a custom autograd function

# %%
import warnings
from typing import Callable, NamedTuple, Optional

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
    f: Callable[tuple[Tensor, Tensor], Tensor],
    x: Tensor,
    z0: Optional[Tensor] = None,
    maxiter: int = 100,
    atol: float = 10**-8,
    rtol: float = 10**-5,
) -> Tensor:
    """Solves $z⁎=f(x，z⁎)$ via FP iteration."""
    if isinstance(f, nn.Module) and z0 is None:
        z = torch.zeros(f.hidden_size)
    elif z0 is None:
        z = torch.zeros_like(x)
    else:
        z = z0

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
# ## 4. Using a custom autograd function - using [Chillee's](https://github.com/Chillee) suggestion


# %%
from functools import wraps

import torch.utils._pytree as pytree


def rpartial(func, /, *fixed_args, **fixed_kwargs):
    r"""Fix positional arguments from the right."""

    @wraps(func)
    def _wrapper(*func_args, **func_kwargs):
        return func(*(func_args + fixed_args), **(func_kwargs | fixed_kwargs))

    return _wrapper


def deq_layer_factory(
    module: nn.Module,
    fsolver=fixed_point_iteration,
    fsolver_kwargs={},
    bsolver=cgs,
    bsolver_kwargs={},
) -> Callable[[Tensor], Tensor]:
    """Create functional deq_layer for the given module."""

    params, param_spec = pytree.tree_flatten({
        **dict(module.named_parameters()), **dict(module.named_buffers())
    })

    fsolver_kwargs = fsolver_kwargs | {"z0": torch.zeros(module.hidden_size)}

    def func(x: Tensor, z: Tensor, *params_and_buffers) -> Tensor:
        """Function call $f(x, z, θ)$."""
        theta = pytree.tree_unflatten(params_and_buffers, param_spec)
        return torch.func.functional_call(module, theta, (x, z))

    class DEQ_Layer(torch.autograd.Function):
        @staticmethod
        def forward(x: Tensor, *params_and_buffers) -> Tensor:
            """Compute the fixed point $z⁎ = f(x, z⁎)$."""
            f = rpartial(func, *params_and_buffers)
            return fsolver(f, x, **fsolver_kwargs)

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, *params_and_buffers = inputs
            zstar = output
            f = rpartial(func, *params_and_buffers)

            with torch.enable_grad():
                # NOTE: without detach, we get an infinite loop.
                zstar = zstar.detach().requires_grad_()
                fstar = f(x, zstar)

            ctx.save_for_backward(fstar, zstar, x)

        @staticmethod
        def backward(ctx, grad_output):
            fstar, zstar, x = ctx.saved_tensors

            # solve the linear system $(𝕀 - ∂f(x，z⁎)/∂z⁎)ᵀg = y$
            L = lambda g: g - grad(fstar, zstar, g, retain_graph=True)[0]
            gstar = bsolver(L, grad_output, **bsolver_kwargs).x

            # compute the outer grads
            grad_x = (
                grad(fstar, x, gstar, retain_graph=True) if x.requires_grad else None
            )
            grad_f = grad(fstar, params, gstar)
            return grad_x, *grad_f

    return rpartial(DEQ_Layer.apply, *params)


# %%
deq_layer = deq_layer_factory(model)

model.zero_grad(set_to_none=True)
y = deq_layer(x).norm().pow(2)
y.backward()

gradients_function = [w.grad for w in model.parameters()]
print("MSE between automatic gradients to manual gradients:")
for g1, g2 in zip(reference_gradients, gradients_function):
    print((g1 - g2).pow(2).mean())

# %% [markdown]
# ## 5. Wrapping layer factory into an `nn.Module`


# %%
class DEQ_Module(nn.Module):
    def __init__(
        self,
        f,
        fsolver=fixed_point_iteration,
        fsolver_kwargs={},
        bsolver=cgs,
        bsolver_kwargs={},
    ) -> None:
        super().__init__()
        self.f = f
        self.fsolver = rpartial(fsolver, **fsolver_kwargs)
        self.bsolver = rpartial(bsolver, **bsolver_kwargs)
        self.deq_layer = deq_layer_factory(
            self.f, fsolver=self.fsolver, bsolver=self.bsolver
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.deq_layer(x)


DEQ = DEQ_Module(model)

DEQ.zero_grad(set_to_none=True)
y = DEQ(x)
y.norm().pow(2).backward()
gradients_module = [w.grad for w in DEQ.parameters()]
print("MSE between automatic gradients to manual gradients:")
for g1, g2 in zip(reference_gradients, gradients_module):
    print((g1 - g2).pow(2).mean())


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

# %%
