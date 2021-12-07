r"""A Lazy Dictionary implementation.

The LazyDict is a dictionary that is initialized with functions as the values.
Once the value is accessed, the function is called and the result is stored.
"""


__all__ = [
    # Classes
    "LazyDict",
    "LazyFunction",
]

import logging
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import Any, NamedTuple, Optional, overload

__logger__ = logging.getLogger(__name__)


class LazyFunction(NamedTuple):
    """A placeholder for uninitialized values."""

    func: Callable
    args: Optional[Iterable] = ()
    kwargs: Optional[Mapping] = {}

    def __call__(self):
        """Execute the function and return the result."""
        return self.func(*self.args, **self.kwargs)

    def __repr__(self):
        """Return a string representation of the function."""
        return f"<LazyFunction: {self.func.__name__}>"


class LazyDict(MutableMapping):
    """A Lazy Dictionary implementation."""

    @staticmethod
    def _validate_value(value: Any) -> LazyFunction:
        r"""Validate the value."""
        if isinstance(value, LazyFunction):
            return value
        if callable(value):
            return LazyFunction(func=value)
        if isinstance(value, tuple):
            if len(value) < 1 or not callable(value[0]):
                raise ValueError("Invalid tuple")
            func = value[0]
            if len(value) == 1:
                return LazyFunction(func)
            if len(value) == 2 and isinstance(value[1], Mapping):
                return LazyFunction(func, kwargs=value[1])
            if len(value) == 2 and isinstance(value[1], Iterable):
                return LazyFunction(func, args=value[1])
            if (
                len(value) == 3
                and isinstance(value[1], Iterable)
                and isinstance(value[2], Mapping)
            ):
                return LazyFunction(func, args=value[1], kwargs=value[2])
            raise ValueError("Invalid tuple")
        raise ValueError("Invalid value")

    @overload
    def __init__(self, /, **kwargs: Any):
        ...

    @overload
    def __init__(self, /, iterable: Iterable, **kwargs: Any):
        ...

    @overload
    def __init__(self, /, mapping: Mapping, **kwargs: Any):
        ...

    def __init__(self, /, *args, **kwargs: Any):
        r"""Initialize the dictionary.

        Tuples of the form (key, (Callable[..., Any), args, kwargs))
        Dict of the form {key: (Callable[..., Any), args, kwargs)}
        """
        self._dict: dict[Any, Any] = {}
        self._initialized: dict[Any, bool] = {}

        if len(args) > 1:
            raise TypeError("Too many positional arguments")

        if len(args) == 0:
            self.update(kwargs)
            return

        arg = args[0]

        if isinstance(arg, Mapping):
            self.update(**arg)
        elif isinstance(arg, Iterable):
            for item in arg:
                if not isinstance(item, tuple) and len(item) == 2:
                    raise ValueError("Invalid iterable")
                key, value = item
                self[key] = value

    def _initialize(self, key):
        """Initialize the key."""
        __logger__.info("%s: Initializing %s", self, key)
        if key not in self._dict:
            raise KeyError(key)
        if not self._initialized[key]:
            self._dict[key] = self._dict[key]()
            self._initialized[key] = True

    def __getitem__(self, key):
        """Get the value of the key."""
        if key not in self._dict:
            raise KeyError(key)
        if not self._initialized[key]:
            value = self._dict[key]
            func, args, kwargs = value
            self._dict[key] = func(*args, **kwargs)
            self._initialized[key] = True
        return self._dict[key]

    def __setitem__(self, key, value):
        """Set the value of the key."""
        self._dict[key] = self._validate_value(value)
        self._initialized[key] = False

    def __delitem__(self, key):
        """Delete the value of the key."""
        del self._dict[key]
        del self._initialized[key]

    def __contains__(self, key):
        """Check if the key is in the dictionary."""
        return key in self._dict

    def __iter__(self):
        """Iterate over the keys."""
        return iter(self._dict)

    def __len__(self):
        """Return the number of keys."""
        return len(self._dict)

    def __repr__(self):
        """Return the representation of the dictionary."""
        padding = " " * 2
        max_key_length = max(len(str(key)) for key in self.keys())
        items = [(str(key), self._dict.get(key)) for key in self]
        maxitems = 10

        string = self.__class__.__name__ + "(\n"
        if maxitems is None or len(self) <= maxitems:
            string += "".join(
                f"{padding}{str(key):<{max_key_length}}: {value}\n"
                for key, value in items
            )
        else:
            string += "".join(
                f"{padding}{str(key):<{max_key_length}}: {value}\n"
                for key, value in items[: maxitems // 2]
            )
            string += f"{padding}...\n"
            string += "".join(
                f"{padding}{str(key):<{max_key_length}}: {value}\n"
                for key, value in items[-maxitems // 2 :]
            )

        string += ")"
        return string

    def __str__(self):
        """Return the string representation of the dictionary."""
        return str(self._dict)
