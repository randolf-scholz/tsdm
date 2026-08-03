r"""Test namedtuple decorator."""

from tsdm.utils.decorators.experimental import return_namedtuple


def test_namedtuple_decorator() -> None:
    @return_namedtuple
    def foo(x: int, y: int) -> tuple[int, int]:
        q, r = divmod(x, y)
        return q, r

    assert str(foo(5, 3)) == "foo_tuple(q=1, r=2)"

    @return_namedtuple(name="divmod")
    def bar(x: int, y: int) -> tuple[int, int]:
        q, r = divmod(x, y)
        return q, r

    assert str(bar(5, 3)) == "divmod(q=1, r=2)"
