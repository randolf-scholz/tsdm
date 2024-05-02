r"""Test performance of match statements with union types."""

from random import choice


def if_else(x):
    if x is None:
        return
    if isinstance(x, bool):
        return
    if isinstance(x, int):
        return
    if isinstance(x, float):
        return
    if isinstance(x, str):
        return
    raise TypeError


def matchit(x):
    match x:
        case None:
            return
        case bool():
            return
        case int():
            return
        case float():
            return
        case str():
            return
        case _:
            raise TypeError


choices = [1, 3.6, True, None, "foo"]

# %%timeit
for _ in range(100_000):  # 29 ms ± 266 µs per loop
    if_else(choice(choices))

# %%timeit
for _ in range(100_000):  # 34.2 ms ± 392 µs per loop
    matchit(choice(choices))


# %% Union
def if_else_union(x):
    if isinstance(x, bool | int | float):
        return
    if isinstance(x, str | bytes):
        return
    raise TypeError


def if_else_tuple(x):
    if isinstance(x, (bool, int, float)):  # noqa: UP038
        return
    if isinstance(x, (str, bytes)):  # noqa: UP038
        return
    raise TypeError


def matchit_union(x):
    match x:
        case bool() | int() | float():
            return
        case str() | bytes():
            return
        case _:
            raise TypeError


choices_union = [1, 3.6, True, "foo", b"bar"]

# %%timeit
for _ in range(100_000):  # 44.6 ms ± 363 µs
    if_else_union(choice(choices_union))

# %%timeit
for _ in range(100_000):  # 34 ms ± 231 µs
    if_else_tuple(choice(choices_union))

# %%timeit
for _ in range(100_000):  # 37.6 ms ± 299 µs
    matchit_union(choice(choices_union))
