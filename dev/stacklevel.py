#!/usr/bin/env python
"""
    test
    Testing the warning stack level usage
    :copyright: (c) 2012 by Openlabs Technologies & Consulting (P) Limited
    :license: BSD, see LICENSE for more details.
"""
import warnings

#: Developer specific warning are usually ignored when run from itnerpreter
#: Force it to be displayed always
warnings.simplefilter("always", DeprecationWarning)


def level_1_warning(not_reqd_arg=None):
    if not_reqd_arg is not None:
        warnings.warn("A level 1 warning", DeprecationWarning, stacklevel=1)
    return True


def level_2_warning(not_reqd_arg=None):
    if not_reqd_arg is not None:
        warnings.warn("A level 2 warning", DeprecationWarning, stacklevel=2)
    return True


def some_function():
    level_1_warning("Nested Level 1")


def some_other_function():
    level_2_warning("Nested Level 2")


if __name__ == "__main__":
    level_1_warning("Not None")
    level_2_warning("Not None")
    some_function()
    some_other_function()
