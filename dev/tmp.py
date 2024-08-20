#!/usr/bin/env python

from typing import NamedTuple

import pytest


class Suite(NamedTuple):
    kind: type
    cases: list[object]


SUITES: dict[str, Suite] = {
    "A": Suite(kind=bool, cases=[True, False]),
    "B": Suite(kind=int, cases=[0, 1, 3]),
    "C": Suite(kind=object, cases=[None]),
}


def has_property(kind: type, item: object) -> bool:
    return True


@pytest.mark.parametrize("suite_name", SUITES)
def test_all(suite_name: str):
    suite = SUITES[suite_name]
    for item in suite:
        assert has_property(suite.kind, item)


# from pytest_cases import fixture, parametrize, parametrize_with_cases
#
#
# @pytest.fixture(params=SUITES)
# def suite_name(request) -> str:
#     return request.param
#
#
# @pytest.fixture
# def suite(suite_name: str) -> Suite:
#     return SUITES[suite_name]
#
#
# def case_element(suite: Suite) -> list[object]:
#     return suite.cases
#
#
# @parametrize_with_cases("element", cases=case_element)
# def test_element(suite, item):
#     assert has_property(suite.kind, item)
