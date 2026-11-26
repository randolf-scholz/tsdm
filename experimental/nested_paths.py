r"""Check filesystem paths in nested data structures."""

__all__ = ["Nested", "nested_paths_exist"]

from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

from tsdm.types import FilePath

type Nested[T] = Mapping[str, Nested[T]] | Collection[Nested[T]] | T  # +T
r"""Type alias for leaves of type `T` nested in string-keyed mappings or collections."""


def nested_paths_exist(paths: Nested[FilePath | None], /) -> bool:
    r"""Check whether the files exist.

    The input may contain arbitrarily nested mappings and iterables with strings
    or `Path` objects as leaves. The check visits mapping values and treats `None`
    and empty containers as having no missing paths.
    """
    match paths:
        case None:
            return True
        case str(string):
            return Path(string).exists()
        case Path() as path:
            return path.exists()
        case Mapping() as mapping:
            return all(nested_paths_exist(path) for path in mapping.values())
        case Iterable() as iterable:
            return all(nested_paths_exist(path) for path in iterable)
        case _:
            raise TypeError(f"Unknown type for rawdata_file: {type(paths)}")
