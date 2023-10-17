"""Experiments with the scaled Lᴾ norm."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


def f(x, p):
    """Normal Lᴾ norm.

    .. Signature:: ``[(..., n), (m,)] -> (..., m)``
    """
    x = np.asarray(x)
    p = np.asarray(p)
    assert p.ndim < 2
    return np.sum(np.power.outer(np.abs(x), p), axis=-2) ** (1 / p)


def g(x, p):
    """Scaled Lᴾ norm.

    .. Signature:: ``[(..., n), (m,)] -> (..., m)``
    """
    x = np.asarray(x)
    p = np.asarray(p)
    assert p.ndim < 2
    return np.mean(np.power.outer(np.abs(x), p), axis=-2) ** (1 / p)


def max_norm(x, p):
    """Maximum norm."""
    return np.max(np.abs(x)) * np.ones_like(p)


def gmean(x):
    """Geometric mean."""
    return np.prod(np.abs(x)) ** (1 / len(x))


def exp_scaled_norm():
    """Plot the scaled Lᴾ norm for different values of p."""
    N = 7
    x = np.random.randn(N)
    p = np.logspace(-3, 3, 1000)

    fig, ax = plt.subplots()

    ax.loglog(p, max_norm(x, p), "--k", lw=2, label="‖x‖ₘₐₓ")
    ax.loglog(p, gmean(x) * np.ones_like(p), "--k", lw=2, label="geometric_mean(x)")
    ax.loglog(p, f(x, p), "-b", lw=2, alpha=0.6, label="‖x‖ₚ ≔ (∑ₙ|xₙ|ᵖ)¹ᐟᵖ")
    ax.loglog(p, g(x, p), "-r", lw=2, alpha=0.6, label="‖x‖ₚ ≔ (⅟ₙ∑ₙ|xₙ|ᵖ)¹ᐟᵖ")
    ax.set_ylim(10**-1, 10**8)
    ax.set_xlabel("p")
    ax.set_ylabel("‖x‖ₚ")
    ax.legend()
    fig.suptitle("Scaled Lₚ norm vs unscaled Lₚ norm.")
    fig.savefig("scaled_norm.png", dpi=500)
    fig.show()


def unit_circle_lp_scaled():
    r"""Plot the unit circle of the scaled Lₚ norm.

    To do this we use polar coordinates. Then we have:

    .. math:: ‖x‖ₚ = c ⟺ |x|ᵖ + |y|ᴾ = cᵖ
        ⟺ |r \cos φ|ᵖ + |r \sin φ|ᴾ = cᵖ
        ⟺ rᵖ |\cos φ|ᵖ + rᵖ |\sin φ|ᴾ = c
        ⟺ rᴾ = cᴾ / (|\cos φ|ᵖ + |\sin φ|ᴾ)
        ⟺ r = c / ‖(\cos φ, \sin φ)‖ₚ

    This equation holds true for both scaled and unscaled version.
    """
    angle = np.linspace(0, 2 * np.pi, 2**16)
    circle = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    p_values = [+4, +2, +1, +0.5, +0.25, -0.25, -0.5, -1, -2, -4]

    norms_normal = f(circle, p_values)
    norms_scaled = g(circle, p_values)

    radius_normal = 1 / norms_normal
    radius_scaled = 1 / norms_scaled

    unit_circle_normal = np.einsum("...n, ...p -> pn...", circle, radius_normal)
    unit_circle_scaled = np.einsum("...n, ...p -> pn...", circle, radius_scaled)

    # use a nicer colormap
    cmap = colormaps["bwr"]
    colors = cmap(np.linspace(1, 0, len(p_values)))

    fig, axes = plt.subplots(ncols=2, figsize=(16, 9), constrained_layout=True)

    for k, p in enumerate(p_values):
        axes[0].plot(*unit_circle_normal[k], lw=2, label=f"p={p:+g}", color=colors[k])
        axes[1].plot(*unit_circle_scaled[k], lw=2, label=f"p={p:+g}", color=colors[k])
        axes[0].set_title("unit circles regular Lᴾ-norm", fontsize=20)
        axes[1].set_title("unit circles scaled  Lᴾ-norm", fontsize=20)
        axes[0].set_aspect("equal")
        axes[1].set_aspect("equal")
        axes[0].set_xlim(-4.5, 4.5)
        axes[0].set_ylim(-4.5, 4.5)
        axes[1].set_xlim(-4.5, 4.5)
        axes[1].set_ylim(-4.5, 4.5)
        axes[0].legend(loc="lower right", fontsize=14)
        axes[1].legend(loc="lower right", fontsize=14)
    # fig.suptitle("Unit circle of the Lₚ norm vs the scaled Lₚ norm.", fontsize=20)
    fig.savefig("unit_circle.png", dpi=300)
    fig.show()
