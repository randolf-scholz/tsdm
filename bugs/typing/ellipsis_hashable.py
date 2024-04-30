from collections.abc import Hashable

x: Hashable = ...
assert isinstance(x, Hashable)


y: type[Hashable] = type(Ellipsis)
