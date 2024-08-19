r"""Tests the attribute decorator."""

import pytest

from tsdm.utils.decorators import attribute


def test_attribute() -> None:
    class Foo:
        @attribute
        def bar(self) -> int:
            return 42

    obj = Foo()
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload is attr.SENTINEL

    # test __get__
    assert obj.bar == 42
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload == 42

    # test __set__
    obj.bar = 24
    assert obj.bar == 24
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload == 24

    # test __delete__
    del obj.bar
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload is attr.DELETED
    with pytest.raises(AttributeError):
        assert obj.bar is attr.DELETED

    # test __set__
    obj.bar = 24
    assert obj.bar == 24
    attr = obj.__class__.__dict__["bar"]
    assert isinstance(attr, attribute)
    assert attr.payload == 24
