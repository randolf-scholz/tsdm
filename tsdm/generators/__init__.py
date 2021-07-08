r"""Generators for synthetic time series datasets.

Contrary to tsdm.datasets, which contains static, real-world datasets, this module
contains generators for synthetic datasets. By design each generators consists of

- Configuration parameters, e.g. number of dimensions etc.
- Allows sampling from the data ground truth distribution p(x,y)
- Allows estimating the Bayes Error, i.e. the best performance possible on the dataset.
"""
# TODO: add some generators
