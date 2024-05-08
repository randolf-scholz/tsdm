r"""Implements the BaseConfig class."""

__all__ = [
    # Classes
    "Config",
    "ConfigMetaclass",
    # Functions
    "is_allcaps",
    "is_dunder",
]

from abc import ABCMeta
from collections.abc import Iterator, MutableMapping
from dataclasses import KW_ONLY, dataclass, field

from typing_extensions import Any, Self


def is_allcaps(s: str) -> bool:
    r"""Check if a string is all caps."""
    return s.isidentifier() and s.isupper() and s.isalpha()


def is_dunder(s: str) -> bool:
    r"""Check if a string is a dunder."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


class ConfigMetaclass(ABCMeta):
    r"""Metaclass for `BaseConfig`."""

    _FORBIDDEN_FIELDS = {
        "clear",       # Removes all the elements from the dictionary
        "copy",        # Returns a copy of the dictionary
        "fromkeys",    # Returns a dictionary with the specified keys and value
        "get",         # Returns the value of the specified key
        "items",       # Returns a list containing a tuple for each key value pair
        "keys",        # Returns a list containing the dictionary's keys
        "pop",         # Removes the element with the specified key
        "popitem",     # Removes the last inserted key-value pair
        "setdefault",  # Returns the value of the specified key or set default
        "update",      # Updates the dictionary with the specified key-value pairs
        "values",      # Returns a list of all the values in the dictionary
    }  # fmt: skip

    # NOTE: This is the canonical signature
    #   https://github.com/python/typeshed/blob/7f9b3ea6c354273ff6ef78c15f274d6e29becb22/stdlib/builtins.pyi#L193-L195
    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> type:
        r"""Create a new class, patch in dataclass fields, and return it."""
        if "__annotations__" not in namespace:
            namespace["__annotations__"] = {}

        config_type = super().__new__(cls, name, bases, namespace, **kwds)
        FIELDS = set(namespace["__annotations__"])

        # check forbidden fields
        FORBIDDEN_FIELDS = cls._FORBIDDEN_FIELDS & FIELDS
        if FORBIDDEN_FIELDS:
            raise ValueError(
                f"Fields {cls._FORBIDDEN_FIELDS!r} are not allowed! "
                f"Found {FORBIDDEN_FIELDS!r}"
            )

        # check for dunder fields
        DUNDER_FIELDS = {key for key in FIELDS if is_dunder(key)}
        if DUNDER_FIELDS:
            raise ValueError(f"Dunder fields are not allowed!Found {DUNDER_FIELDS!r}.")

        # check all caps fields
        ALLCAPS_FIELDS = {key for key in FIELDS if is_allcaps(key)}
        if ALLCAPS_FIELDS:
            raise ValueError(f"ALLCAPS fields are reserved!Found {ALLCAPS_FIELDS!r}.")

        NAME = config_type.__qualname__.rsplit(".", maxsplit=1)[0]
        patched_fields = [
            ("_", KW_ONLY),
            ("NAME", str, field(default=NAME)),
            ("MODULE", str, field(default=namespace["__module__"])),
        ]

        for key, hint, *value in patched_fields:
            config_type.__annotations__[key] = hint
            if value:
                setattr(config_type, key, value[0])

        return dataclass(config_type, eq=False, frozen=True)  # type: ignore[call-overload]


class Config(MutableMapping[str, Any], metaclass=ConfigMetaclass):
    r"""Base Config."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the config."""
        self.update(*args, **kwargs)

    def __iter__(self) -> Iterator[str]:
        r"""Return an iterator over the keys of the dictionary."""
        return iter(self.__dict__)

    def __getitem__(self, key: str, /) -> Any:
        r"""Return the value of the specified key."""
        return self.__dict__[key]

    def __len__(self) -> int:
        r"""Return the number of items in the dictionary."""
        return len(self.__dict__)

    def __hash__(self) -> int:
        r"""Returns permutation-invariant hash on `items()`."""
        return hash(frozenset(self.items()))

    def __or__(self, other: dict) -> Self:
        r"""Return a new dictionary with the keys from both dictionaries."""
        res: dict = {}
        res.update(self)
        res.update(other)
        return self.__class__(**res)

    def __setitem__(self, key, value, /):
        self.__dict__[key] = value

    def __delitem__(self, key, /):
        del self.__dict__[key]
