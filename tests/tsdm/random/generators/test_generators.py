r"""Test synthetic generators."""

import matplotlib.pyplot as plt
import numpy as np
from pytest import mark

from tsdm.config import PROJECT
from tsdm.random.generators import (
    SIR,
    BouncingBall,
    DampedPendulum,
    DampedPendulumXY,
    LotkaVolterra,
)

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@mark.flaky(reruns=3)
def test_bouncing_ball() -> None:
    """Test Bouncing Ball."""
    # sample from generator
    t = np.linspace(-10, 20, 256)
    y = BouncingBall().rvs(t)
    # generate plot
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(t, y, ".", label="location")
    ax.set_xlabel("time t")
    ax.set_ylabel("location x")
    ax.set_title("Time Series Plot")
    ax.legend()

    # save plot
    fig.savefig(RESULT_DIR / "bouncing_ball.png")


@mark.flaky(reruns=3)
def test_lotka_volterra() -> None:
    """Test Lotka-Volterra."""
    # sample from generator
    t = np.linspace(0, 20, 256)
    y = LotkaVolterra().rvs(t)

    # generate plot
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(t, y[..., 0], ".", label="prey")
    axes[0].plot(t, y[..., 1], ".", label="predator")
    axes[0].set_xlabel("time t")
    axes[0].set_title("Time Series Plot")
    axes[0].legend()
    axes[1].plot(y[:, 0], y[:, 1], ".")
    axes[1].set_title("Phase plot")

    # save plot
    fig.savefig(RESULT_DIR / "lotka_volterra.png")


@mark.flaky(reruns=3)
def test_damped_pendulum() -> None:
    """Test Damped Pendulum."""
    # sample from generator
    t = np.linspace(0, 10, 256)
    y = DampedPendulum().rvs(t)

    # generate plot
    colors = iter(plt.colormaps["tab10"].colors)  # type: ignore[attr-defined]
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(t, y[..., 0], ".", label="θ", color=next(colors))
    axes[0].set_ylim(-max(abs(y[..., 0])), +max(abs(y[..., 0])))
    axes_0y = axes[0].twinx()
    axes_0y.plot(t, y[..., 1], ".", label="ω", color=next(colors))
    axes_0y.set_ylim(-max(abs(y[..., 1])), +max(abs(y[..., 1])))
    axes[0].set_ylabel("angle $θ$")
    axes_0y.set_ylabel("angular velocity $ω$")
    axes[0].set_xlabel("time t")
    axes[0].set_title("Time Series Plot")
    axes[0].legend()
    axes[1].plot(y[:, 0], y[:, 1], ".", color=next(colors))
    axes[1].set_title("Phase plot")

    # save plot
    fig.savefig(RESULT_DIR / "damped_pendulum.png")


@mark.xfail(reason="batching not supported by scipy solve_ivp")
def test_damped_pendulum_batch() -> None:
    t = np.linspace(0, 10, 128)
    num_sequences = 3
    y = DampedPendulum().rvs(t, size=(num_sequences,))
    assert y.shape == (num_sequences, t.size, 2)


@mark.flaky(reruns=3)
def test_damped_pendulum_xy() -> None:
    """Test Damped Pendulum XY."""
    # sample from generator
    t = np.linspace(0, 10, 512)
    y = DampedPendulumXY().rvs(t)

    # generate plot
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(t, y[..., 0], ".", label="x")
    axes[0].plot(t, y[..., 1], ".", label="y")
    axes[0].set_xlabel("time t")
    axes[0].set_title("Time Series Plot")
    axes[0].legend()
    axes[1].plot(y[:, 0], y[:, 1], ".")
    axes[1].set_title("Phase plot")
    axes[1].set_aspect("equal", "box")

    # save plot
    fig.savefig(RESULT_DIR / "damped_pendulum_xy.png")


@mark.flaky(reruns=3)
def test_sir_model() -> None:
    """Test SIR model."""
    # sample from generator
    t = np.linspace(0, 100, 256)
    y = SIR(alpha=0.1, beta=0.5).rvs(t)

    # generate plot
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(t, y[..., 0], ".", label="S")
    axes[0].plot(t, y[..., 1], ".", label="I")
    axes[0].plot(t, y[..., 2], ".", label="R")
    axes[0].set_xlabel("time t")
    axes[0].set_title("Time Series Plot")
    axes[0].legend()
    axes[1].set_xlabel("Infected")
    axes[1].set_ylabel("Recovered")
    axes[1].plot(y[:, 1], y[:, 2], ".")
    axes[1].set_title("Phase plot")

    # save plot
    fig.savefig(RESULT_DIR / "sir_model.png")
