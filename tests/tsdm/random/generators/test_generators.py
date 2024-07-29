r"""Test synthetic generators."""

from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from tsdm.config import PROJECT
from tsdm.random.generators import (
    SIR,
    BouncingBall,
    DampedPendulum,
    DampedPendulumXY,
    LotkaVolterra,
)

RESULT_DIR = PROJECT.RESULTS_DIR[__file__]


@pytest.mark.xfail(reason="batching not supported by scipy solve_ivp")
def test_damped_pendulum_batch() -> None:
    t = np.linspace(0, 10, 128)
    num_sequences = 3
    y = DampedPendulum().rvs(t, size=(num_sequences,))
    assert y.shape == (num_sequences, t.size, 2)


@pytest.mark.flaky(reruns=3)
def test_bouncing_ball() -> None:
    r"""Test Bouncing Ball."""
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
    fig.suptitle(f"Bouncing Ball (generated {datetime.now()})")

    # save plot
    fig.savefig(RESULT_DIR / "bouncing_ball.png")


@pytest.mark.flaky(reruns=3)
def test_lotka_volterra() -> None:
    r"""Test Lotka-Volterra."""
    # sample from generator
    t = np.linspace(0, 20, 256)
    y = LotkaVolterra().rvs(t)

    # generate plot
    ax0: Axes
    ax1: Axes
    fig, [[ax0, ax1]] = plt.subplots(
        ncols=2, figsize=(12, 6), constrained_layout=True, squeeze=False
    )
    ax0.plot(t, y[..., 0], ".", label="prey")
    ax0.plot(t, y[..., 1], ".", label="predator")
    ax0.set_xlabel("time t")
    ax0.set_title("Time Series Plot")
    ax0.legend()
    ax1.plot(y[:, 0], y[:, 1], ".")
    ax1.set_title("Phase plot")
    fig.suptitle(f"Lotka-Volterra Model (generated {datetime.now()})")

    # save plot
    fig.savefig(RESULT_DIR / "lotka_volterra.png")


@pytest.mark.flaky(reruns=3)
def test_damped_pendulum() -> None:
    r"""Test Damped Pendulum."""
    # sample from generator
    t = np.linspace(0, 10, 256)
    y = DampedPendulum().rvs(t)

    # generate plot
    colors = iter(plt.colormaps["tab10"].colors)  # type: ignore[attr-defined]
    ax0: Axes
    ax1: Axes
    fig, [[ax0, ax1]] = plt.subplots(
        ncols=2, figsize=(12, 6), constrained_layout=True, squeeze=False
    )

    ax0y: Axes = ax0.twinx()  # type: ignore[assignment]
    ax0y.plot(t, y[..., 1], ".", label="ω", color=next(colors))
    ax0y.set_ylim(-max(abs(y[..., 1])), +max(abs(y[..., 1])))
    ax0y.set_ylabel("angular velocity $ω$")

    ax0.plot(t, y[..., 0], ".", label="θ", color=next(colors))
    ax0.set_ylim(-max(abs(y[..., 0])), +max(abs(y[..., 0])))
    ax0.set_ylabel("angle $θ$")
    ax0.set_xlabel("time t")
    ax0.set_title("Time Series Plot")
    ax0.legend()

    ax1.plot(y[:, 0], y[:, 1], ".", color=next(colors))
    ax1.set_title("Phase plot")
    fig.suptitle(f"Dampled Pendulum (generated {datetime.now()})")

    # save plot
    fig.savefig(RESULT_DIR / "damped_pendulum.png")


@pytest.mark.flaky(reruns=3)
def test_damped_pendulum_xy() -> None:
    r"""Test Damped Pendulum XY."""
    # sample from generator
    t = np.linspace(0, 10, 512)
    y = DampedPendulumXY().rvs(t)

    # generate plot
    ax0: Axes
    ax1: Axes
    fig, [[ax0, ax1]] = plt.subplots(
        ncols=2, figsize=(12, 6), constrained_layout=True, squeeze=False
    )
    ax0.plot(t, y[..., 0], ".", label="x")
    ax0.plot(t, y[..., 1], ".", label="y")
    ax0.set_xlabel("time t")
    ax0.set_title("Time Series Plot")
    ax0.legend()
    ax1.plot(y[:, 0], y[:, 1], ".")
    ax1.set_title("Phase plot")
    ax1.set_aspect("equal", "box")
    fig.suptitle(f"Dampled Pendulum XY (generated {datetime.now()})")

    # save plot
    fig.savefig(RESULT_DIR / "damped_pendulum_xy.png")


@pytest.mark.flaky(reruns=3)
def test_sir_model() -> None:
    r"""Test SIR model."""
    # sample from generator
    t = np.linspace(0, 100, 256)
    y = SIR(alpha=0.1, beta=0.5).rvs(t)

    # generate plot
    ax0: Axes
    ax1: Axes
    fig, [[ax0, ax1]] = plt.subplots(
        ncols=2, figsize=(12, 6), constrained_layout=True, squeeze=False
    )
    ax0.plot(t, y[..., 0], ".", label="S")
    ax0.plot(t, y[..., 1], ".", label="I")
    ax0.plot(t, y[..., 2], ".", label="R")
    ax0.set_xlabel("time t")
    ax0.set_title("Time Series Plot")
    ax0.legend()
    ax1.set_xlabel("Infected")
    ax1.set_ylabel("Recovered")
    ax1.plot(y[:, 1], y[:, 2], ".")
    ax1.set_title("Phase plot")
    fig.suptitle(f"SIR Model (generated {datetime.now()})")

    # save plot
    fig.savefig(RESULT_DIR / "sir_model.png")
