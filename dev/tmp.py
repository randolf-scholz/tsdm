#!/usr/bin/env python


class A:
    foo = f"{__package__}.{__module__}.{__name__}.{__qualname__}"


print(A.foo)
