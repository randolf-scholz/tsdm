r"""Logging Utility Functions."""

__all__ = [
    # Submodules
    "callbacks",
    "logfuncs",
    "loggers",
    "utils",
    # Constants
    "LOGGERS",
    "LOGFUNCS",
    "CALLBACKS",
]

from . import callbacks, logfuncs, loggers, utils
from .callbacks import *  # ruff: ignore[F403]
from .logfuncs import *  # ruff: ignore[F403]
from .loggers import *  # ruff: ignore[F403]
from .utils import *  # ruff: ignore[F403]

__all__ += callbacks.__all__
__all__ += logfuncs.__all__
__all__ += loggers.__all__
__all__ += utils.__all__


CALLBACKS: dict[str, type[callbacks.Callback]] = {
    "CallbackList"        : callbacks.CallbackList,
    "CheckpointCallback"  : callbacks.CheckpointCallback,
    "ConfigCallback"      : callbacks.ConfigCallback,
    "EvaluationCallback"  : callbacks.EvaluationCallback,
    "HParamCallback"      : callbacks.HParamCallback,
    "KernelCallback"      : callbacks.KernelCallback,
    "LRSchedulerCallback" : callbacks.LRSchedulerCallback,
    "MetricsCallback"     : callbacks.MetricsCallback,
    "ModelCallback"       : callbacks.ModelCallback,
    "OptimizerCallback"   : callbacks.OptimizerCallback,
    "ScalarsCallback"     : callbacks.ScalarsCallback,
    "TableCallback"       : callbacks.TableCallback,
    "WrapCallback"        : callbacks.WrapCallback,
}  # fmt: skip
r"""Dictionary of all available callbacks."""


LOGFUNCS: dict[str, logfuncs.LogFunction] = {
    "log_config"       : logfuncs.log_config,
    "log_kernel"       : logfuncs.log_kernel,
    "log_lr_scheduler" : logfuncs.log_lr_scheduler,
    "log_metrics"      : logfuncs.log_metrics,
    "log_model"        : logfuncs.log_model,
    "log_optimizer"    : logfuncs.log_optimizer,
    "log_plot"         : logfuncs.log_plot,
    "log_table"        : logfuncs.log_table,
    "log_values"       : logfuncs.log_values,
}  # fmt: skip
r"""Dictionary of all available log functions."""

LOGGERS: dict[str, type[loggers.Logger]] = {
    "BaseLogger"    : loggers.BaseLogger,
    "DefaultLogger" : loggers.DefaultLogger,
}  # fmt: skip
r"""Dictionary of all available loggers."""
