#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: kiwi
#     language: python
#     name: python3
# ---

# %%
# %config InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.
# %config InlineBackend.figure_format = 'svg'
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import logging

logging.basicConfig(level=logging.INFO)

# %%
import tsdm

# %%
ds = tsdm.datasets.PhysioNet2012()

# %%
set(ds.timeseries_description.lower_bound.keys())

# %%

# %%
set.intersection({1, 2}, [2, 4])

# %%
tsdm.datasets.timeseries.USHCN()

# %%
set.intersection({"a, b"}, ["a", "c"], ["a", "d"])

# %%
set.intersection({0, 1}, [0, 2], [0, 3])

# %%
dicts = [{"foo": 0, "bar": 1, "baz": 2}, {"foo": 3, "bar": 4}, {"foo": 5, "baz": 6}]
set.intersection(*(set(d) for d in dicts))
set.intersection(*dicts)

# %%
set.union(dicts)

# %%
from functools import partial, reduce


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        if instance is None:
            return super().__get__(instance, type_)
        return partial(super().__get__(instance, type_), instance)


class Set(set):
    @class_or_instancemethod
    def intersection(cls, first, *other):
        return reduce(cls.__and__, (cls(first), *map(cls, other)))


assert Set.intersection([0, 1], [0, 2], [0, 3]) == {0}  # {0}
assert Set.intersection([0, 1]) == {0, 1}  # {0,1}

assert Set([0, 1]).intersection([0, 2], [0, 3]) == {0}  # {0}
Set([0, 1]).intersection() == {
    0,
    1,
}  # 1 missing 1 required positional argument: 'first'

# %%
Set([0, 1]).intersection([0, 2], [0, 3])


# %%

# %%


# %%
class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        print(f"{self=} {instance=} {type_=}")

        # if instance is not None:
        #     return super().__get__(instance, type_)

        # return self.__func__.__get__

        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


class A:
    @class_or_instancemethod
    def foo(cls, first, *other):
        print(f"{cls=}")
        print(f"{first=}")
        print(f"{other=}")
        ret = [first, *other]
        print(f"{ret=}")
        return ret


# %%
A.foo(...)
A.foo(1, 2, 3)

# %%
A().foo(...)

# %%
