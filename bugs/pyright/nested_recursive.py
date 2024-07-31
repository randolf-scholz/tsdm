type Nested[T] = T | list["Nested[T]"]


class Transform[X, Y]:
    def __invert__(self) -> "Transform[Y, X]": ...


class ListTransform[X, Y](Transform[Nested[X], Nested[Y]]):
    def __invert__(self) -> "ListTransform[Y, X]": ...
