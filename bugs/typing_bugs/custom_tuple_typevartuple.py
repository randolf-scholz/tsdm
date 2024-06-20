from typing_extensions import Generic, TypeVarTuple, reveal_type

Ts = TypeVarTuple("Ts")


class MyTuple(Generic[*Ts]):
    items: tuple[*Ts]

    def __init__(self, *args: *Ts):
        self.items = args

    def __reversed__(self) -> "MyTuple[*Ts[::-1]]":
        return MyTuple(*reversed(self.items))

    def __repr__(self):
        return f"MyTuple{self.items}"


my_tup = MyTuple(1, "A", 3.14)
reveal_type(my_tup)
my_tup_reversed = reversed(my_tup)
reveal_type(my_tup_reversed)
print(my_tup_reversed)
