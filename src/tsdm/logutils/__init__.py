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
    # ABCs & Protocols
    "BaseCallback",
    "BaseLogger",
    "Callback",
    "CallbackSequence",
    "DefaultLogger",
    "LogFunction",
    "Logger",
    # Classes
    "CallbackList",
    "CheckpointCallback",
    "ConfigCallback",
    "EvaluationCallback",
    "HParamCallback",
    "KernelCallback",
    "LRSchedulerCallback",
    "MetricsCallback",
    "ModelCallback",
    "OptimizerCallback",
    "ScalarsCallback",
    "TableCallback",
    "WrapCallback",
    # Functions
    "is_callback",
    "is_logfunc",
    "log_config",
    "log_kernel",
    "log_lr_scheduler",
    "log_metrics",
    "log_model",
    "log_optimizer",
    "log_plot",
    "log_table",
    "log_values",
    # extras
    "save_checkpoint",
]

from . import callbacks, logfuncs, loggers, utils
from .callbacks import (
    BaseCallback,
    Callback,
    CallbackList,
    CallbackSequence,
    CheckpointCallback,
    ConfigCallback,
    EvaluationCallback,
    HParamCallback,
    KernelCallback,
    LRSchedulerCallback,
    MetricsCallback,
    ModelCallback,
    OptimizerCallback,
    ScalarsCallback,
    TableCallback,
    WrapCallback,
    is_callback,
)
from .logfuncs import (
    LogFunction,
    is_logfunc,
    log_config,
    log_kernel,
    log_lr_scheduler,
    log_metrics,
    log_model,
    log_optimizer,
    log_plot,
    log_table,
    log_values,
)
from .loggers import BaseLogger, DefaultLogger, Logger
from .utils import save_checkpoint

CALLBACKS: dict[str, type[Callback]] = {
    "CallbackList"        : CallbackList,
    "CheckpointCallback"  : CheckpointCallback,
    "ConfigCallback"      : ConfigCallback,
    "EvaluationCallback"  : EvaluationCallback,
    "HParamCallback"      : HParamCallback,
    "KernelCallback"      : KernelCallback,
    "LRSchedulerCallback" : LRSchedulerCallback,
    "MetricsCallback"     : MetricsCallback,
    "ModelCallback"       : ModelCallback,
    "OptimizerCallback"   : OptimizerCallback,
    "ScalarsCallback"     : ScalarsCallback,
    "TableCallback"       : TableCallback,
    "WrapCallback"        : WrapCallback,
}  # fmt: skip
r"""Dictionary of all available callbacks."""


LOGFUNCS: dict[str, LogFunction] = {
    "log_config"       : log_config,
    "log_kernel"       : log_kernel,
    "log_lr_scheduler" : log_lr_scheduler,
    "log_metrics"      : log_metrics,
    "log_model"        : log_model,
    "log_optimizer"    : log_optimizer,
    "log_plot"         : log_plot,
    "log_table"        : log_table,
    "log_values"       : log_values,
}  # fmt: skip
r"""Dictionary of all available log functions."""

LOGGERS: dict[str, type[Logger]] = {
    "BaseLogger"    : BaseLogger,
    "DefaultLogger" : DefaultLogger,
}  # fmt: skip
r"""Dictionary of all available loggers."""
