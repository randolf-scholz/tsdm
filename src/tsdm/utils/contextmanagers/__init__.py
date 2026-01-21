r"""Context managers."""

__all__ = [
    # CONSTANTS
    "CONTEXT_MANAGERS",
    # classes
    "ContextManager",
    "system_path",
    "ray_cluster",
    "timer",
    "timeout",
]

from tsdm.utils.contextmanagers._contextmanagers import (
    ContextManager,
    ray_cluster,
    system_path,
    timeout,
    timer,
)

CONTEXT_MANAGERS: dict[str, type[ContextManager]] = {
    "add_to_path" : system_path,
    "ray_cluster" : ray_cluster,
    "timeout"     : timeout,
    "timer"       : timer,
}  # fmt: skip
r"""Dictionary of all available context managers."""
