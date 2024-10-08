#  SEE: https://github.com/microsoft/pyright/issues/5849#issuecomment-1708441977

from typing import ClassVar, NamedTuple, Protocol


class MyNamedTuple(NamedTuple):
    stuff: int


class NamedTupleProtocol(Protocol):
    _fields: ClassVar[tuple[str, ...]]


def show_fields(t: type[NamedTupleProtocol]) -> None:
    print(t._fields)  # OK


# works for pyright 1.1.334 to 1.1.341
show_fields(MyNamedTuple)  # KO

# Argument of type "Type[Thing]" cannot be assigned to parameter "t" of type "Type[TA@demoA]" in function "demoA"
#   Type "Thing" cannot be assigned to type "NamedTuplishA"
#     "Thing" is incompatible with protocol "NamedTuplishA"
#       "_fields" is not a class variable (lsp)
