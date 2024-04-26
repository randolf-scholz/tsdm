r"""Logging Utility Functions."""

__all__ = [
    # Submodules
    "callbacks",
    "logfuncs",
    "loggers",
    # Constants
    "LOGGERS",
    "LOGFUNCS",
    "CALLBACKS",
    # ABCs & Protocols
    "Logger",
    "Callback",
    "LogFunction",
    "BaseLogger",
    "BaseCallback",
    # Classes
    "CheckpointCallback",
    "ConfigCallback",
    "DefaultLogger",
    "EvaluationCallback",
    "HParamCallback",
    "KernelCallback",
    "LRSchedulerCallback",
    "MetricsCallback",
    "ModelCallback",
    "OptimizerCallback",
    "ScalarsCallback",
    "TableCallback",
    # Functions
    "make_checkpoint",
    "log_kernel",
    "log_lr_scheduler",
    "log_metrics",
    "log_model",
    "log_optimizer",
    "log_table",
    "log_values",
]

from tsdm.logutils import callbacks, logfuncs, loggers
from tsdm.logutils.callbacks import (
    BaseCallback,
    Callback,
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
)
from tsdm.logutils.logfuncs import (
    LogFunction,
    log_kernel,
    log_lr_scheduler,
    log_metrics,
    log_model,
    log_optimizer,
    log_table,
    log_values,
    make_checkpoint,
)
from tsdm.logutils.loggers import BaseLogger, DefaultLogger, Logger

CALLBACKS: dict[str, type[Callback]] = {
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
}  # fmt: skip
r"""Dictionary of all available callbacks."""


log_kernel2: LogFunction = log_kernel
log_lr_scheduler2: LogFunction = log_lr_scheduler
log_metrics2: LogFunction = log_metrics
log_model2: LogFunction = log_model
log_optimizer2: LogFunction = log_optimizer
log_values2: LogFunction = log_values
log_table2: LogFunction = log_table


LOGFUNCS: dict[str, LogFunction] = {
    "log_kernel"       : log_kernel,
    "log_lr_scheduler" : log_lr_scheduler,
    "log_metrics"      : log_metrics,
    "log_model"        : log_model,
    "log_optimizer"    : log_optimizer,
    "log_values"       : log_values,
    "log_table"        : log_table,
}  # fmt: skip
r"""Dictionary of all available log functions."""

LOGGERS: dict[str, type[Logger]] = {
    "BaseLogger"    : BaseLogger,
    "DefaultLogger" : DefaultLogger,
}  # fmt: skip
r"""Dictionary of all available loggers."""
