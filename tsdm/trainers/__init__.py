r"""
Trainers are more generic than optimizers.
They include the whole training loop, i.e. also stopping criteria, learn-rate schedulers and so on.
"""

from tsdm.trainers.lr_schedulers import LR_SCHEDULERS

__all__ = ["LR_SCHEDULERS"]
