r"""Checks the internal consistency of the module constants."""

from collections.abc import Callable, Mapping
from inspect import isabstract
from types import ModuleType
from typing import NamedTuple

import pytest
from typing_extensions import is_protocol

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
from tsdm.metrics import (
    FUNCTIONAL_LOSSES,
    MODULAR_LOSSES,
    TIMESERIES_LOSSES,
    BaseMetric,
    Metric,
    TimeSeriesBaseLoss,
    TimeSeriesLoss,
)
from tsdm.models import MODELS, BaseModel, ForecastingModel
from tsdm.optimizers import (
    LR_SCHEDULERS,
    OPTIMIZERS,
    LRScheduler,
    Optimizer,
    TorchLRScheduler,
    TorchOptimizer,
)
from tsdm.random.generators import GENERATORS, IVP_Generator, IVP_GeneratorBase
from tsdm.random.samplers import SAMPLERS, BaseSampler, Sampler
from tsdm.utils.decorators import (
    CLASS_DECORATORS,
    FUNCTION_DECORATORS,
    ClassDecorator,
    FunctionDecorator,
)


class Case(NamedTuple):
    r"""NamedTuple for each case."""

    module: ModuleType
    protocol: type
    base_class: type | None
    elements: Mapping[str, type] | Mapping[str, Callable]


CASES: dict[str, Case] = {
    "callbacks"      : Case(tsdm.logutils          , Callback          , BaseCallback       , CALLBACKS           ),
    "datasets"       : Case(tsdm.datasets          , Dataset           , BaseDataset        , DATASETS            ),
    "encoders"       : Case(tsdm.encoders          , Encoder           , BaseEncoder        , ENCODERS            ),
    "generators"     : Case(tsdm.random.generators , IVP_Generator     , IVP_GeneratorBase  , GENERATORS          ),
    "loggers"        : Case(tsdm.logutils          , Logger            , BaseLogger         , LOGGERS             ),
    "lr_schedulers"  : Case(tsdm.optimizers        , LRScheduler       , TorchLRScheduler   , LR_SCHEDULERS       ),
    "metrics     "   : Case(tsdm.metrics           , Metric            , BaseMetric         , MODULAR_LOSSES      ),
    "metrics_time"   : Case(tsdm.metrics           , TimeSeriesLoss    , TimeSeriesBaseLoss , TIMESERIES_LOSSES   ),
    "models"         : Case(tsdm.models            , ForecastingModel  , BaseModel          , MODELS              ),
    "optimizers"     : Case(tsdm.optimizers        , Optimizer         , TorchOptimizer     , OPTIMIZERS          ),
    "samplers"       : Case(tsdm.random.samplers   , Sampler           , BaseSampler        , SAMPLERS            ),
    "logfuncs"       : Case(tsdm.logutils.logfuncs , LogFunction       , None               , LOGFUNCS            ),
    "decorators_cls" : Case(tsdm.utils.decorators  , ClassDecorator    , None               , CLASS_DECORATORS    ),
    "decorators_fun" : Case(tsdm.utils.decorators  , FunctionDecorator , None               , FUNCTION_DECORATORS ),
    "metrics_fun"    : Case(tsdm.metrics           , Metric            , None               , FUNCTIONAL_LOSSES   ),
}  # fmt: skip
r"""Dictionary of all available cases."""


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    # REF: https://stackoverflow.com/q/40818146
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
    # fallback for jit.ScriptFunction
    name = getattr(obj, "__name__", getattr(obj, "name", None))
    assert name == item_name


def test_issubclass(case_name: str, item_name: str) -> None:
    r"""Check if the class is a subclass of the correct base class."""
    case = CASES[case_name]
    obj = case.elements[item_name]

    if case.base_class is not None:
        assert issubclass(obj, case.base_class)  # type: ignore[arg-type]


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
