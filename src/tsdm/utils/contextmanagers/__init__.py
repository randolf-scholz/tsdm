r"""Context managers."""

__all__ = [
    # CONSTANTS
    "CONTEXT_MANAGERS",
    # classes
    "system_path",
    "ray_cluster",
    "timer",
    "timeout",
]

from contextlib import AbstractContextManager

from tsdm.utils.contextmanagers._contextmanagers import (
    ray_cluster,
    system_path,
    timeout,
    timer,
)

CONTEXT_MANAGERS: dict[str, type[AbstractContextManager]] = {
    "add_to_path": system_path,
    "ray_cluster": ray_cluster,
    "timer": timer,
    "timeout": timeout,
}
r"""Dictionary of all available context managers."""

del AbstractContextManager
