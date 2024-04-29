"""Context managers."""

__all__ = [
    # CONSTANTS
    "CONTEXT_MANAGERS",
    # classes
    "add_to_path",
    "ray_cluster",
    "timer",
]

from contextlib import AbstractContextManager

from tsdm.utils.contextmanagers._contextmanagers import add_to_path, ray_cluster, timer

CONTEXT_MANAGERS: dict[str, type[AbstractContextManager]] = {
    "add_to_path": add_to_path,
    "ray_cluster": ray_cluster,
    "timer": timer,
}
r"""Dictionary of all available context managers."""

del AbstractContextManager
