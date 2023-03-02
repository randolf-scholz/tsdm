r"""Logging Utility Functions."""

__all__ = [
    # Types
    "Logger",
    "Callback",
    # Constants
    "LOGGERS",
    "CALLBACKS",
    # Functions
    "compute_metrics",
    "log_optimizer",
    "log_kernel",
    "log_lr_scheduler",
    "log_model",
    "log_metrics",
    "log_scalars",
    "log_table",
    # Classes
    "StandardLogger",
    "BaseLogger",
    "DefaultLogger",
    # Callbacks
    "BaseCallback",
    "CheckpointCallback",
    "KernelCallback",
    "LRSchedulerCallback",
    "MetricsCallback",
    "ModelCallback",
    "OptimizerCallback",
    "ScalarsCallback",
    "TableCallback",
]

from tsdm.logutils._callbacks import (
    BaseCallback,
    Callback,
    CheckpointCallback,
    KernelCallback,
    LRSchedulerCallback,
    MetricsCallback,
    ModelCallback,
    OptimizerCallback,
    ScalarsCallback,
    TableCallback,
    compute_metrics,
    log_kernel,
    log_lr_scheduler,
    log_metrics,
    log_model,
    log_optimizer,
    log_scalars,
    log_table,
)
from tsdm.logutils.base import BaseLogger, DefaultLogger, Logger, StandardLogger

CALLBACKS: dict[str, Callback] = {
    "CheckpointCallback": CheckpointCallback,
    "KernelCallback": KernelCallback,
    "LRSchedulerCallback": LRSchedulerCallback,
    "MetricsCallback": MetricsCallback,
    "ModelCallback": ModelCallback,
    "OptimizerCallback": OptimizerCallback,
    "ScalarsCallback": ScalarsCallback,
    "TableCallback": TableCallback,
}

LOGGERS: dict[str, Logger] = {
    "log_kernel_information": log_kernel,
    "log_lr_scheduler": log_lr_scheduler,
    "log_metrics": log_metrics,
    "log_model_state": log_model,
    "log_optimizer_state": log_optimizer,
    "log_scalars": log_scalars,
    "log_table": log_table,
}
