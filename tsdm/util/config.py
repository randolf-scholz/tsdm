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

# __all__ = []


__logger__ = logging.getLogger(__name__)
