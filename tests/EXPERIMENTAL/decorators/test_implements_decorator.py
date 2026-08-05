from tsdm.types.callbacks import IdentityMapOnCls


def implements(*protocols: type) -> IdentityMapOnCls:
    r"""Check if class implements a set of protocols."""

    def __wrapper[Cls: type](cls: Cls, /) -> Cls:
        if __debug__:
            for protocol in protocols:
                if not issubclass(cls, protocol):
                    raise TypeError(f"{cls} does not implement {protocol}")
        return cls

    return __wrapper
