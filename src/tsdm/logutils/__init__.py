r"""Logging Utility Functions."""

__all__ = [
    # Submodules
    "callbacks",
    "logfuncs",
    "loggers",
    # Types
    "Logger",
    "Callback",
    "LogFunction",
    # Constants
    "LOGGERS",
    "LOGFUNCS",
    "CALLBACKS",
    # Functions
    "make_checkpoint",
    "log_kernel",
    "log_lr_scheduler",
    "log_metrics",
    "log_model",
    "log_optimizer",
    "log_table",
    "log_values",
    # Classes
    "BaseLogger",
    "DefaultLogger",
    "BaseCallback",
    # Callbacks
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
    "CheckpointCallback": CheckpointCallback,
    "ConfigCallback": ConfigCallback,
    "EvaluationCallback": EvaluationCallback,
    "HParamCallback": HParamCallback,
    "KernelCallback": KernelCallback,
    "LRSchedulerCallback": LRSchedulerCallback,
    "MetricsCallback": MetricsCallback,
    "ModelCallback": ModelCallback,
    "OptimizerCallback": OptimizerCallback,
    "ScalarsCallback": ScalarsCallback,
    "TableCallback": TableCallback,
}

LOGFUNCS: dict[str, LogFunction] = {
    "log_kernel": log_kernel,
    "log_lr_scheduler": log_lr_scheduler,
    "log_metrics": log_metrics,
    "log_model": log_model,
    "log_optimizer": log_optimizer,
    "log_values": log_values,
    "log_table": log_table,
}

LOGGERS: dict[str, type[Logger]] = {
    "BaseLogger": BaseLogger,
    "DefaultLogger": DefaultLogger,
}
