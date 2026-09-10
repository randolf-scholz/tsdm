r"""Loss functions that by definition include aggregation over samples.

This includes for instance the standard RMSE loss.
These should not be evaluated on mini-batches, instead they should only be estimated
on the full dataset.
"""
