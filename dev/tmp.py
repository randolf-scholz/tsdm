#!/usr/bin/env python

import numpy as np

a = np.float64(4)
b = np.float64(5)
max(a, b)


class Foo:
    def __lt__(self, other):
        return self

    def __gt__(self, other):
        return self

    def __le__(self, other):
        return self

    def __ge__(self, other):
        return self


x = Foo()
y = Foo()
assert min(x, y) is y
