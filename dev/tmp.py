class Foo[X, Y, Z = object](Base[X, Y]):
    def test(self, swapped: Foo[Y, X]) -> None:
        reveal_type(swapped)  # "Foo[X`1, Y`2, builtins.object]" ❓️❗️
