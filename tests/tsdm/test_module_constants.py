r"""Checks the internal consistency of the module constants."""

from collections.abc import Callable, Mapping
from inspect import isabstract
from types import ModuleType
from typing import NamedTuple, is_protocol

import pytest

import tsdm
from tsdm.datasets import DATASETS, BaseDataset, Dataset
from tsdm.encoders import ENCODERS, BaseEncoder, Encoder
from tsdm.logutils import (
    CALLBACKS,
    LOGFUNCS,
    LOGGERS,
    BaseCallback,
    BaseLogger,
    Callback,
    LogFunction,
    Logger,
)
from tsdm.losses import (
    FUNCTIONAL_LOSSES,
    MODULAR_LOSSES,
    TIMESERIES_LOSSES,
    BaseLoss,
    Loss,
)
from tsdm.losses.base import BaseSequenceLoss, SequenceLoss
from tsdm.random.generators import GENERATORS, BaseIVP_Generator, IVP_Generator
from tsdm.random.samplers import SAMPLERS, BaseSampler, Sampler


class Case(NamedTuple):
    r"""NamedTuple for each case."""

    module: ModuleType
    protocol: type
    base_class: type | None
    elements: Mapping[str, type] | Mapping[str, Callable]


CASES: dict[str, Case] = {
    "callbacks"      : Case(tsdm.logutils          , Callback          , BaseCallback       , CALLBACKS         ),
    "datasets"       : Case(tsdm.datasets          , Dataset           , BaseDataset        , DATASETS          ),
    "encoders"       : Case(tsdm.encoders          , Encoder           , BaseEncoder        , ENCODERS          ),
    "generators"     : Case(tsdm.random.generators , IVP_Generator     , BaseIVP_Generator  , GENERATORS        ),
    "logfuncs"       : Case(tsdm.logutils.logfuncs , LogFunction       , None               , LOGFUNCS          ),
    "loggers"        : Case(tsdm.logutils          , Logger            , BaseLogger         , LOGGERS           ),
    "metrics     "   : Case(tsdm.losses            , Loss              , BaseLoss           , MODULAR_LOSSES    ),
    "metrics_fun"    : Case(tsdm.losses            , Loss              , None               , FUNCTIONAL_LOSSES ),
    "metrics_time"   : Case(tsdm.losses            , SequenceLoss      , BaseSequenceLoss   , TIMESERIES_LOSSES ),
    "samplers"       : Case(tsdm.random.samplers   , Sampler           , BaseSampler        , SAMPLERS          ),
}  # fmt: skip
r"""Dictionary of all available cases."""


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    # SEE: https://stackoverflow.com/q/40818146
    # FIXME: https://github.com/pytest-dev/pytest/issues/349
    # FIXME: https://github.com/pytest-dev/pytest/issues/4050
    if "case_name" in metafunc.fixturenames:
        if "item_name" in metafunc.fixturenames:
            metafunc.parametrize(
                ["case_name", "item_name"],
                [
                    (name, element)
                    for name, case in CASES.items()
                    for element in case.elements
                ],
            )
        else:
            metafunc.parametrize("case_name", CASES)


def test_protocol(case_name: str) -> None:
    case = CASES[case_name]
    assert is_protocol(case.protocol)


def test_base_class(case_name: str) -> None:
    case = CASES[case_name]
    cls = case.base_class

    if cls is not None:
        assert isinstance(cls, type)
        assert not is_protocol(cls)


def test_name(case_name: str, item_name: str) -> None:
    case = CASES[case_name]
    obj = case.elements[item_name]
    basename = item_name.rsplit(".", maxsplit=1)[-1]
    class_name = getattr(obj, "__name__", None)
    # fallback for jit.ScriptFunction
    fallback_name = getattr(obj, "name", None)
    assert basename in {class_name, fallback_name}


def test_issubclass(case_name: str, item_name: str) -> None:
    r"""Check if the class is a subclass of the correct base class."""
    case = CASES[case_name]
    obj = case.elements[item_name]

    if case.base_class is not None:
        assert isinstance(obj, type)
        assert issubclass(obj, case.base_class)


def test_dict_complete(case_name: str) -> None:
    r"""Check if all encoders are in the ENCODERS constant."""
    case = CASES[case_name]

    match case.base_class:
        case None:  # skip for function-dicts
            return
        case base_class:
            missing: dict[str, type] = {
                name: cls
                for name, cls in vars(case.module).items()
                if (
                    not is_protocol(cls)
                    and not isabstract(cls)
                    and isinstance(cls, type)
                    and issubclass(cls, base_class)
                    and cls is not base_class
                )
                and cls not in case.elements.values()
            }

    if missing:
        raise AssertionError(f"Missing {case.protocol.__name__}: {sorted(missing)}")
