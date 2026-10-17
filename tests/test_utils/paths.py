r"""Paths for persistent test artifacts."""

__all__ = ["get_results_dir"]

from os import PathLike
from pathlib import Path


def get_results_dir(file: str | PathLike[str], /) -> Path:
    r"""Create and return ``tests/.results/<file stem>`` for a test module.

    Locate the enclosing ``tests`` directory whose parent contains a ``.git``
    directory or worktree file. Relative paths are resolved against the working
    directory. Files with the same stem share a results directory. Existing
    contents are preserved on repeated calls.

    Raises:
        ValueError: If the file is not under ``<git_root>/tests``.
    """
    file_path = Path(file).resolve()
    for parent in file_path.parents:
        if parent.name == "tests" and (parent.parent / ".git").exists():
            path = parent / ".results" / file_path.stem
            path.mkdir(parents=True, exist_ok=True)
            return path
    raise ValueError(f"Expected a file under <git_root>/tests, got {file_path}")
