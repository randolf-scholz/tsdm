r"""Test for copy.replace with __replace__ method."""

import copy
from typing import Any, Self


def test_copy_replace() -> None:
    class Foo:
        def __replace__(self, *arg: Any, **kwargs: Any) -> Self:
            print(f"__replace__ called with \n\t{arg=}\n\t{kwargs=}")
            return self

    foo = Foo()
    copy.replace(foo)


def test_copy_replace_classmethod() -> None:
    class Foo:
        @classmethod
        def __replace__(cls, *arg: Any, **kwargs: Any) -> Self:
            print(f"__replace__ called with \n\t{arg=}\n\t{kwargs=}")
            return cls()

    foo = Foo()
    copy.replace(foo)
