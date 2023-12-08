#!/usr/bin/env python

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    # mypy throws a used-before-def error despite `from __future__ import annotations`
    FunctionOfMyClass = Callable[[MyClass], None]


class MyClass:
    # This (only) works due to `from __future__ import annotations`
    def compare(self, other: MyClass) -> None:
        pass


def function_of_my_class(my_class: MyClass) -> None:
    print(my_class)


def function_of_function_of_my_class(f_of_my_myclass: FunctionOfMyClass) -> None:
    print(f_of_my_myclass)


function_of_function_of_my_class(function_of_my_class)
