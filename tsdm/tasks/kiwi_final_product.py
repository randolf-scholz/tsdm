r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    # 'KIWI_FINAL_PRODUCT',
]

import logging
import os
from copy import deepcopy
from functools import cached_property
from itertools import product
from typing import Any, Literal, Mapping, Optional, Sequence

import torch
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, Timestamp, Timedelta
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from tsdm.datasets import KIWI_RUNS, Dataset
from tsdm.datasets.torch.generic import DatasetCollection
from tsdm.encoders.modular import (
    BaseEncoder,
    ChainedEncoder,
    DataFrameEncoder,
    DateTimeEncoder,
    FloatEncoder,
    MinMaxScaler,
    Standardizer,
    TensorEncoder,
)
from tsdm.losses.modular import WRMSE
from tsdm.random.samplers import CollectionSampler, SequenceSampler
from tsdm.tasks.base import BaseTask
from util.types import KeyType


def get_induction_time(s: Series) -> Timestamp:
    # Compute the induction time
    # s = ts.loc[run_id, exp_id]
    inducer = s["InducerConcentration"]
    total_induction = inducer[-1] - inducer[0]

    if pd.isna(total_induction) or total_induction == 0:
        return pd.NA

    diff = inducer.diff()
    mask = pd.notna(diff) & (diff != 0.0)
    inductions = inducer[mask]
    assert len(inductions) == 1, "Multiple Inductions occur!"
    return inductions.first_valid_index()


def get_final_product(s: Series, target) -> Timestamp:
    # Final and target times
    targets = s[target]
    mask = pd.notna(targets)
    targets = targets[mask]
    assert len(targets) >= 1, f"not enough target observations {targets}"
    return targets.index[-1]


def get_time_table(ts: DataFrame, target="Fluo_GFP", t_min="0.6h", delta_t="5m"):
    columns = [
        "slice",
        "t_min",
        "t_induction",
        "t_max",
        "t_target",
    ]
    index = ts.reset_index(level=[2]).index.unique()
    df = DataFrame(index=index, columns=columns)

    min_wait = Timedelta(t_min)

    for idx, slc in ts.groupby(level=[0, 1]):
        slc = slc.reset_index(level=[0, 1], drop=True)
        # display(slc)
        t_induction = get_induction_time(slc)
        t_target = get_final_product(slc, target=target)
        if pd.isna(t_induction):
            print(f"{idx}: no t_induction!")
            t_max = get_final_product(slc.loc[slc.index < t_target], target=target)
            assert t_max < t_target
        else:
            assert t_induction < t_target, f"{t_induction=} after {t_target}!"
            t_max = t_induction
        df.loc[idx, "t_max"] = t_max

        df.loc[idx, "t_min"] = t_min = slc.index[0] + min_wait
        df.loc[idx, "t_induction"] = t_induction
        df.loc[idx, "t_target"] = t_target
        df.loc[idx, "slice"] = slice(t_min, t_max)
        # = t_final
    return df
#
#
# def get_final_vector(ts, md, target="Fluo_GFP", t_min="0.6h", delta_t="5m"):
#
#
#
#
#
# class KIWI_FINAL_PRODUCT(BaseTask):
#
#
#     def __init__(self,) -> None:
#         super().__init__()
#
#
#         dataset =
#
#
#         final_product_times = get_final_product(s, target="Fluo_GFP")
#
#         # Compute the final vector
#         final_vecs = {}
#
#         for idx in md.index:
#             t_target = final_product_times.loc[idx, "t_target"]
#             final_vecs[(*idx, t_target)] = ts.loc[idx].loc[t_target]
#
#         final_vec = DataFrame.from_dict(final_vec, orient="index")
#         final_vec.index = final_vec.index.set_names(ts.index.names)
#         final_vec = final_vec[target]







    # @property
    # def index(self) -> Sequence[KeyType]:
    #     pass
    #
    # @property
    # def splits(self) -> Mapping[KeyType, Any]:
    #     pass
    #
    # def get_dataloader(self, key: KeyType, *, batch_size: int = 1, shuffle: bool = False,
    #                    device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    #                    **kwargs: Any) -> DataLoader:
    #     pass
    #
    # ...



