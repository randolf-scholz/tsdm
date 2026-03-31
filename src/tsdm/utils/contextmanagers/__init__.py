r"""Context managers."""

__all__ = [
    # CONSTANTS
    "CONTEXT_MANAGERS",
    # classes
    "ContextManager",
    "system_path",
    "ray_cluster",
    "timer",
]

from ._contextmanagers import ContextManager, ray_cluster, system_path, timer

CONTEXT_MANAGERS: dict[str, type[ContextManager]] = {
    "add_to_path" : system_path,
    "ray_cluster" : ray_cluster,
    "timer"       : timer,
}  # fmt: skip
r"""Dictionary of all available context managers."""
