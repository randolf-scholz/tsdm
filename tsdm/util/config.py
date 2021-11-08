r"""General purpose configuration.

Goal
~~~~
We want a config class to have a direct 1:1 correspondence between
nested dicts and config files. We want to be able to write

.. code-block::

    Config(
      num_layers = 10,
      activation = 'relu',
      optimizer = Config(
         __type__ =  "Adam",
         lr = 0.01,
         beta1 = 0.99,
         beta2 = 0.999,
      ),
    )

Features
~~~~~~~~
- permutation invariant hash: recursively ``sum( hash((key, hash(value))) )``
- hash equivalent to hash dict/config file after serialization?
- __iter__ -> to_dict
- __set__
- __bool__
- __str__
- __repr__
- __ror__, __ior__, __or__ combine keys, overwrite from right
- __rand__, __iand__, __and__ intersect keys, overwrite from right
- __ge__, __le__, __eq__, __neq__
- __rsub__, __isub__, __sub__ remove keys from the right
- # __radd__, __iadd__, __add__ add keys from the right, keep left
- __len__
- __contains__
- __getitem__, __setitem__, __delitem__
- register as MutableMapping: https://docs.python.org/3/library/collections.abc.html
- to_dict
- to_json
- to_yaml
- to_toml
- to_string
- allow comments? __comments__: dict[key, str] will be written as comments after serialization?
- Type Hint protocol - https://github.com/python/mypy/issues/731

**How config should work**
1. Decide on BaseTypes (bool, int, float, str, ...)
2. Decide on ContainerTypes (list, tuple, set, ...)
3. Decide how ContainerTypes can be nested (tuple[list], list[tuple], etc.)
4. Specify type recursively.

**Example**
1. BaseTypes = bool, int, float, str
2. BaseContainers = list
3. ContainerTypes = list[ContainerTypes] | Union[list[t] for t in BaseTypes]
4. JSON = dict[str, JSON | BaseTypes | ContainerTypes]

**Implementation**
- Config = Dataclass factory (with extras)
- do not allow dunder-keys except specified namespace

Second part
-----------
Create a helper function "initialize_from"

.. code-block:: python

    def initialize_from(DICT: dict[str, OBJ], conf: dict or Config) -> Callable
        cls = conf["__name__"]
        kwargs = {k:v for k,v in conf.items() if k != "__name__"}
        if isclass(obj):
           result = obj(**kwargs)
       elif callable(obj):
           result = partial(obj, **kwargs)
       else:
            raise value_error
        return result
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Optional, Union

__all__ = ["Config"]

__logger__ = logging.getLogger(__name__)


def is_dunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


class Config(Iterable):
    """Remark - Initializing dicts.

    There are 3 ways of initializing `dict`

    - `dict(**kwargs)`: standard key/values
    - `dict(Mapping, **kwargs)`:  If a mapping object is given, then
        1. A list of keys `list[key]` will be generated via `list(iter(Mapping))`
        2. The values will be looked up via `Mapping.__getitem__(key)`
    - `dict(Iterable, **kwargs)`: If the first item is an iterable, then:
        1. A `list[tuple[key, value]]` will be generated via `list(iter(Iterable))`
    """

    def __init__(
        self, __dict__: Optional[Union[Mapping, Iterable]] = None, /, **kwargs
    ):
        super().__init__()

        if __dict__ is not None:
            assert not kwargs, f"kwargs not allowed if Mappping given!"

        if isinstance(__dict__, Mapping):
            for key in __dict__:
                value = __dict__[key]
                if isinstance(value, Config):
                    self._add_key_value(key, value)
                elif is_dunder(key):
                    raise ValueError(f"Cannot set dunder key {key=}")
                # Recurse on Mapping
                self._add_key_value(key, value)
        elif isinstance(__dict__, Iterable):
            items = list(__dict__)
            for item in items:
                assert isinstance(item, Iterable)
                tup = tuple(item)
                assert len(tup) == 2, f"Too many inputs."
                key, value = tup
                self._add_key_value(key, value)
        else:
            raise ValueError(f"Data Type {type(__dict__)} not understood!")

        for key, value in kwargs.items():
            self._add_key_value(key, value)

    def _add_key_value(self, key: str, value):
        assert isinstance(key, str), f"Only string keys allowed"
        assert not hasattr(self, key), f"key already taken!"
        if isinstance(value, Mapping):
            setattr(self, key, Config(value))
        else:
            setattr(self, key, value)

    def __repr__(self, nest_level: int = 0):
        print(nest_level)
        pad = r"_" * 4
        start_string = f"{self.__class__.__name__}("
        end_string = f")"

        lines = [start_string]

        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                s = pad + f"{key} = {value.__repr__(nest_level + 1)}"
            else:
                s = pad + f"{key} = {value}"
            lines.append(s)
        lines.append(end_string)
        result = ("\n" + pad * nest_level).join(lines)
        # print(result)
        return result

    def __len__(self):
        return self.__dict__.__len__()

    def __getitem__(self, key, from_iter=False):
        print(f"__getitem__ called from {id(self)} with {key=} and {from_iter=}")
        value = self.__dict__[key]

        if from_iter and isinstance(value, Config):
            return dict(value)
        return value

    def __iter__(self):
        print(f"__iter__ called, {id(self)=}")
        print(f"{self.__dict__=}")
        for key, value in self.__dict__.items():
            # if isinstance(value, Config):
            yield key, self.__getitem__(key, from_iter=True)
