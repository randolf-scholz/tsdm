#!/usr/bin/env python


import numpy as np
from numpy.typing import NDArray


class A: ...


def foo() -> NDArray[A]:  # ✅ raises type-var
    return np.array([A()])


def bar() -> list[NDArray[A]]:  # ❌ does not raise type-var
    return [np.array([A()])]
