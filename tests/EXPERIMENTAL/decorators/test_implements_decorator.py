from tsdm.types.callbacks import Polymorphism


def implements(*protocols: type) -> Polymorphism:
    r"""Check if class implements a set of protocols."""

    def __wrapper[Cls: type](cls: Cls, /) -> Cls:
        if __debug__:
            for protocol in protocols:
                if not issubclass(cls, protocol):
                    raise TypeError(f"{cls} does not implement {protocol}")
        return cls

    return __wrapper
