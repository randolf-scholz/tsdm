#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: kiwi
#     language: python
#     name: python3
# ---

# %%
# %config InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.
# %config InlineBackend.figure_format = 'svg'
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import logging

logging.basicConfig(level=logging.INFO)

# %%
import tsdm

# %%
tsdm.datasets.timeseries.USHCN()

# %%
print(repr("a"))

# %%
