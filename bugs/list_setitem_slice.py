#!/usr/bin/env python


import pytest

if __name__ == "__main__":
    with pytest.raises(IndexError):
        items = [0, 1, 2, 3]
        items[len(items) + 1] = -1

    items = [0, 1, 2, 3]
    items[len(items) + 1 :] = [-1]
    assert items == [0, 1, 2, 3, -1]

    items = [0, 1, 2, 3]
    items[len(items) + 42 :] = [-1]
    assert items == [0, 1, 2, 3, -1]

    items = [0, 1, 2, 3]
    items[len(items) + 42 : len(items) + 45] = [-1]
    assert items == [0, 1, 2, 3, -1]

    items = [0, 1, 2, 3]
    items[-17:-1] = [-1]
    assert items == [-1, 3]

    items = [0, 1, 2, 3]
    items[-17:0] = [-1]
    assert items == [-1, 0, 1, 2, 3]

    items = [0, 1, 2, 3]
    items[-17:1] = [-1]
    assert items == [-1, 1, 2, 3]
