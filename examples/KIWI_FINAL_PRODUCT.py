#!/usr/bin/env python
# coding: utf-8


import argparse

parser = argparse.ArgumentParser(prog="PROG")
parser.add_argument("target", type=str, help="Either OD600 or Fluo_GFP", default="OD600")
parser.add_argument("--split", type=int, help="0, 1, 2, 3 or 4", default=0)
args = parser.parse_args()

TARGET = args.target
SPLIT = args.split

assert TARGET in ["OD600", "Fluo_GFP"], f"{args.target=}"

print(f"{TARGET=}")
# In[1]:

import logging
import os
import sys
from pathlib import Path
from time import perf_counter, time
from typing import Any, NamedTuple

# enable JIT compilation - must be done before loading torch!
os.environ["PYTORCH_JIT"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# logging.basicConfig(level=logging.INFO)


def header(s: str, pad=3):
    n = (79 - len(s)) // 2 - pad
    s = s.upper().center(79 - 2 * n)
    print(f"\n{'>'*n + s + '<'*n}\n", file=sys.stdout, flush=True)


# In[2]:


import torch
import torchinfo

BATCH_SIZE = 128
# TARGET = "OD600"

available_gpus = {i: torch.cuda.device(i) for i in range(torch.cuda.device_count())}
print(f"Availabel GPUS: {available_gpus=}")
# assert len(available_gpus)==0
DEVICE = next(iter(available_gpus.values()))
print(f"{DEVICE=}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE=}")


RUN_NAME = f"{TARGET}-{SPLIT}-More_params"  # | input("enter name for run")


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas
from linodenet.models import LinODE, LinODECell, LinODEnet
from linodenet.models.filters import SequentialFilter
from linodenet.projections.functional import skew_symmetric, symmetric
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from torch import Tensor, jit, nn, tensor
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm, trange

import tsdm
from tsdm.datasets import DATASETS
from tsdm.encoders.functional import time2float
from tsdm.encoders.modular import *
from tsdm.logutils import (
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
)
from tsdm.losses import LOSSES
from tsdm.tasks import KIWI_FINAL_PRODUCT
from tsdm.util import grad_norm, multi_norm
from tsdm.util.strings import *

# In[26]:


# Disable benchmarking for variable sized input
torch.backends.cudnn.benchmark = False


# # Initialize Task

# In[5]:


DTYPE = torch.float32

task = KIWI_FINAL_PRODUCT(
    train_batch_size=BATCH_SIZE,
    eval_batch_size=2048,
    target=TARGET,
)

DATASET = task.dataset
ts = task.timeseries
md = task.metadata
NUM_PTS, NUM_DIM = ts.shape


# ## Initialize Encoder

# In[6]:


ts, md = task.splits[SPLIT, "train"]


encoder = ChainedEncoder(
    TensorEncoder(names=("time", "value", "index")),
    DataFrameEncoder(
        column_encoders={
            "value": IdentityEncoder(),
            tuple(ts.columns): FloatEncoder("float32"),
        },
        index_encoders=MinMaxScaler() @ DateTimeEncoder(unit="h"),
    ),
    TripletEncoder(sparse=True),
    Standardizer(),
)
encoder.fit(ts.reset_index([0, 1], drop=True))
task.target_idx = task.timeseries.columns.get_loc(task.target)
target_encoder = TensorEncoder() @ FloatEncoder() @ encoder[-1][task.target_idx]


# ## Define Batching Function

# In[7]:


class Batch(NamedTuple):
    index: Tensor
    timeseries: Tensor
    metadata: Tensor
    targets: Tensor
    encoded_targets: Tensor

    def __repr__(self):
        return repr_mapping(
            self._asdict(), title=self.__class__.__name__, repr_fun=repr_array
        )


@torch.no_grad()
def mycollate(batch: list):
    index = []
    timeseries = []
    metadata = []
    targets = []
    encoded_targets = []

    for idx, (ts_data, (md_data, target)) in batch:
        index.append(torch.tensor(idx[0]))
        timeseries.append(encoder.encode(ts_data))
        metadata.append(md_data)
        targets.append(target)
        encoded_targets.append(target_encoder.encode(target))

    index = torch.stack(index)
    targets = pandas.concat(targets)
    encoded_targets = torch.concat(encoded_targets)

    return Batch(index, timeseries, metadata, targets, encoded_targets)


# ## Initialize Loss & Metrics

# In[8]:


LOSS = task.test_metric.to(device=DEVICE)
metrics = {key: jit.script(LOSSES[key]()) for key in ("RMSE", "MSE", "MAE")}


# ## Initialize DataLoaders

# In[28]:


TRAINLOADER = task.get_dataloader(
    (SPLIT, "train"),
    batch_size=BATCH_SIZE,
    collate_fn=mycollate,
    pin_memory=True,
    drop_last=True,
    shuffle=True,
    num_workers=8,
    # num_workers=os.cpu_count() // 4,
    persistent_workers=True,
)


EVALLOADER = task.get_dataloader(
    (SPLIT, "test"),
    batch_size=BATCH_SIZE,
    collate_fn=mycollate,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
    num_workers=8,
    # num_workers=os.cpu_count() // 4,
    persistent_workers=True,
)


# ## Hyperparamters

# In[10]:


OPTIMIZER_CONFIG = {
    "__name__": "AdamW",
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0.001,
    "amsgrad": False,
}

LR_SCHEDULER_CONFIG = {
    "__name__": "ReduceLROnPlateau",
    "mode": "min",
    # (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
    "factor": 0.1,
    # (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
    "patience": 10,
    # (int) – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
    "threshold": 0.0001,
    # (float) – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
    "threshold_mode": "rel",
    # (str) – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
    "cooldown": 0,
    # (int) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
    "min_lr": 1e-08,
    # (float or list) – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
    "eps": 1e-08,
    # (float) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    "verbose": True
    # (bool) – If True, prints a message to stdout for each update. Default: False.
}


# ## Model

# In[11]:

###############################################################################
header("INITIALIZE MODEL")  #
#############################

from tsdm.models import SetFuncTS

MODEL = SetFuncTS
model = MODEL(17, 1, latent_size=256, dim_keys=128, dim_vals=128, dim_deepset=128)
model.to(device=DEVICE, dtype=DTYPE)
summary(model)


# ### Warmup - test forward / backward pass

# In[12]:

###############################################################################
header("FORWARD PASS TEST")  #
##############################

batch = next(iter(TRAINLOADER))
print("Loaded batch")
y = model.forward_batch(batch.timeseries)
print("forward done")
torch.linalg.norm(y).backward()
print("backward done")
model.zero_grad()

# In[13]:
###############################################################################
header("INITIALIZE OPTIMIZER ")  #
##################################

from tsdm.optimizers import LR_SCHEDULERS, OPTIMIZERS
from tsdm.util import initialize_from

OPTIMIZER_CONFIG |= {"params": model.parameters()}
optimizer = initialize_from(OPTIMIZERS, **OPTIMIZER_CONFIG)

# In[16]:
###############################################################################
header("INITIALIZE LOGGING ")  #
################################

from tsdm.logutils import compute_metrics

RUN_START = tsdm.util.now()
CHECKPOINTDIR = Path(
    f"checkpoints/{MODEL.__name__}/{DATASET.name}/{RUN_NAME}/{RUN_START}"
)
CHECKPOINTDIR.mkdir(parents=True, exist_ok=True)
LOGGING_DIR = f"runs/{MODEL.__name__}/{DATASET.name}/{RUN_NAME}/{RUN_START}"
writer = SummaryWriter(LOGGING_DIR)


@torch.no_grad()
def get_all_predictions(model, dataloader):
    ys = []
    yhats = []
    for batch in dataloader:
        # ts = batch.timeseries
        # inputs = [(t.to(device=DEVICE),v.to(device=DEVICE), m.to(device=DEVICE)) for t,v,m in ts]
        # yhats.append(model.batch_forward(inputs))
        yhats.append(model.forward_batch(batch.timeseries))
        ys.append(batch.encoded_targets.to(device=DEVICE))
    y = torch.cat(ys)
    yhat = torch.cat(yhats)
    y = torch.tensor(target_encoder.decode(y))
    yhat = torch.tensor(target_encoder.decode(yhat))
    return y, yhat


# In[19]:
###############################################################################
header("EVALUATE INITAILIZED MODEL LOGGING ")  #
################################################

i = -1
epoch = 1

with torch.no_grad():
    for key, dloader in {"train": TRAINLOADER, "test": EVALLOADER}.items():
        y, ŷ = get_all_predictions(model, dloader)
        assert torch.isfinite(y).all()
        log_metrics(epoch, writer, metrics=metrics, targets=y, predics=ŷ, prefix=key)

# In[ ]:
###############################################################################
header("BEGIN TRAINING ")  #
############################

for epoch in (epochs := range(epoch, 2000)):
    if epoch == 1000:
        for g in optimizer.param_groups:
            g["lr"] = 0.0001

    batching_time = perf_counter()
    for batch in TRAINLOADER:
        batching_time = perf_counter() - batching_time
        i += 1
        # Optimization step
        model.zero_grad(set_to_none=True)
        targets = batch.encoded_targets.to(device=DEVICE)

        # forward
        forward_time = perf_counter()
        predics = model.forward_batch(batch.timeseries)
        loss = LOSS(targets, predics)
        forward_time = perf_counter() - forward_time

        # backward
        backward_time = perf_counter()
        loss.backward()
        backward_time = perf_counter() - backward_time

        # step
        optimizer.step()

        # batch logging
        with torch.no_grad():
            logging_time = time()
            if torch.any(~torch.isfinite(loss)):
                raise RuntimeError("NaN/INF-value encountered!!")

            log_metrics(
                i,
                writer,
                metrics=metrics,
                targets=targets.clone(),
                predics=predics.clone(),
                prefix="batch",
            )
            log_optimizer_state(i, writer, optimizer, prefix="batch")

            # lval = loss.clone().detach().cpu().numpy()
            # gval = grad_norm(list(model.parameters())).clone().detach().cpu().numpy()
            logging_time = time() - logging_time

        print(
            dict(
                # loss=f"{lval:.2e}",
                # gnorm=f"{gval:.2e}",
                epoch=epoch,
                Δt_forward=f"{forward_time:.1f}",
                Δt_backward=f"{backward_time:.1f}",
                Δt_logging=f"{logging_time:.1f}",
                Δt_batching=f"{batching_time:.1f}",
            )
        )
        batching_time = perf_counter()

    with torch.no_grad():
        # end-of-epoch logging
        for key, dloader in {"train": TRAINLOADER, "test": EVALLOADER}.items():
            y, ŷ = get_all_predictions(model, dloader)
            assert torch.isfinite(y).all()
            log_metrics(
                epoch, writer, metrics=metrics, targets=y, predics=ŷ, prefix=key
            )

        # Model Checkpoint
        torch.jit.save(model, CHECKPOINTDIR.joinpath(f"{MODEL.__name__}-{epoch}"))
        torch.save(
            {
                "optimizer": optimizer,
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "batch": i,
            },
            CHECKPOINTDIR.joinpath(f"{optimizer.__class__.__name__}-{epoch}"),
        )
