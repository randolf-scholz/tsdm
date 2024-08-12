def deco(cls):
    print("decorating", cls)
    return cls


class Meta(type):
    def __call__(cls, *args, **kwargs):
        print("Meta.__call__")
        return super().__call__(*args, **kwargs)


class Base:
    def __new__(cls, *args, **kwargs):
        print("Base.__new__")
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        print("Base.__init__")

    def __init_subclass__(cls, **kwargs):
        print("Base.__init_subclass__")


@deco
class A(Base):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        print("A.__new__")
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__()
        print("A.__init__")


print("finalized class")

A()
