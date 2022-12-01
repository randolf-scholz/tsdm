#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # KIWI

# %% [markdown]
# ## Input Parsing (for command line use)

# %%
import argparse

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for MIMIC dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=100,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=64,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=64,     type=int,   help="hidden-size")
parser.add_argument("-ls", "--latent-size",  default=128,    type=int,   help="latent-size")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-kp", "--kernel-param", default="identity",         help="kernel-parametrization")
parser.add_argument("-fc", "--filter",       default="SequentialFilter", help="filter-component")
parser.add_argument("-ec", "--encoder",      default="ResNet",           help="encoder-component")
parser.add_argument("-dc", "--decoder",      default="ResNet",           help="decoder-component")
parser.add_argument("-sc", "--system",       default="LinODECell",       help="system-component")
parser.add_argument("-eb", "--embedding",    default="ConcatEmbedding",  help="embedding-component")
parser.add_argument("-pc", "--projection",   default="ConcatProjection", help="projection-component")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
# fmt: on

try:
    get_ipython().run_line_magic(
        "config", "InteractiveShell.ast_node_interactivity='last_expr_or_assign'"
    )
except NameError:
    ARGS = parser.parse_args()
else:
    ARGS = parser.parse_args("")

print(ARGS)

# %% [markdown]
# ### Load from config file if provided

# %%
import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)

# %% [markdown]
# ## Global Variables

# %%
import logging
import os
import pickle
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit
from tqdm.autonotebook import tqdm, trange

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_start_method('spawn')

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")

# %% [markdown]
# ## Initialize Task

# %%
from tsdm.tasks import KiwiTask

TASK = KiwiTask()

# %% [markdown]
# ## Hyperparameter choices

# %%
if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

hparam_dict = {
    "dataset": (DATASET := "KIWI"),
    "model": (MODEL_NAME := "LinODEnet"),
    "fold": ARGS.fold,
    "seed": ARGS.seed,
    "max_epochs": ARGS.epochs,
    "batch_size": ARGS.batch_size,
    "hidden_size": ARGS.hidden_size,
    "latent_size": ARGS.latent_size,
    "kernel-initialization": ARGS.kernel_init,
} | OPTIMIZER_CONFIG


# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_STR = f"f={ARGS.fold}_bs={ARGS.batch_size}_lr={ARGS.learn_rate}_hs={ARGS.hidden_size}_ls={ARGS.latent_size}"
RUN_ID = ARGS.run_id or datetime.now().isoformat(timespec="seconds")
CFG_ID = 0 if ARGS.config is None else ARGS.config[1]
HOME = Path.cwd()

LOGGING_DIR = HOME / "tensorboard" / DATASET / MODEL_NAME / RUN_ID / CONFIG_STR
CKPOINT_DIR = HOME / "checkpoints" / DATASET / MODEL_NAME / RUN_ID / CONFIG_STR
RESULTS_DIR = HOME / "results" / DATASET / MODEL_NAME / RUN_ID
LOGGING_DIR.mkdir(parents=True, exist_ok=True)
CKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Initialize DataLoaders

# %%
# from tsdm.tasks.mimic_iii_debrouwer2019 import mimic_collate as task_collate_fn

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    # "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": os.cpu_count() * 3 // 4,
    # "collate_fn": task_collate_fn,
}

dloader_config_infer = {
    "batch_size": 2 * ARGS.batch_size,
    # "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": os.cpu_count() * 3 // 4,
    # "collate_fn": task_collate_fn,
}

# TRAIN_LOADER = TASK.dataloaders[ARGS.fold, "train"]
# INFER_LOADER = TASK.dataloaders[ARGS.fold, "train"]
# VALID_LOADER = TASK.dataloaders[ARGS.fold, "valid"]
# TEST_LOADER = TASK.dataloaders[ARGS.fold, "test"]

TRAIN_LOADER = TASK.make_dataloader((ARGS.fold, "train"), **dloader_config_train)
INFER_LOADER = TASK.make_dataloader((ARGS.fold, "train"), **dloader_config_infer)
VALID_LOADER = TASK.make_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
TEST_LOADER = TASK.make_dataloader((ARGS.fold, "test"), **dloader_config_infer)
EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}

# %% [markdown]
# # Serialize Encoder

# %%
ENCODER = TASK.encoders[ARGS.fold, "train"]

with open(CKPOINT_DIR / "encoder.pickle", "wb") as file:
    pickle.dump(ENCODER, file)

# check loading
with open(CKPOINT_DIR / "encoder.pickle", "rb") as file:
    _ = pickle.load(file)

# %% [markdown]
# ## Initialize Loss

# %%
from tsdm.metrics import MAE, MSE, RMSE

LOSS = TASK.test_metrics[ARGS.fold, "train"]
METRICS = {
    "time_MSE": LOSS,
}

LOSS = LOSS.to(device=DEVICE)

# %% [markdown]
# ## Initialize Model

# %%
from linodenet.models import LinODEnet
from linodenet.models.embeddings import EMBEDDINGS
from linodenet.models.encoders import ENCODERS
from linodenet.models.filters import FILTERS
from linodenet.models.system import SYSTEMS

MODEL_CONFIG = {
    "__name__": "LinODEnet",
    "input_size": TASK.dataset.timeseries.shape[-1],
    "hidden_size": ARGS.hidden_size,
    "latent_size": ARGS.latent_size,
    "Filter": FILTERS[ARGS.filter].HP | {"autoregressive": True},
    "System": SYSTEMS[ARGS.system].HP
    | {
        "kernel_initialization": ARGS.kernel_init,
        "kernel_parametrization": ARGS.kernel_param,
    },
    "Encoder": ENCODERS[ARGS.encoder].HP | {"num_blocks": 10},
    "Decoder": ENCODERS[ARGS.decoder].HP | {"num_blocks": 10},
    "Embedding": EMBEDDINGS[ARGS.embedding].HP,
    "Projection": EMBEDDINGS[ARGS.projection].HP,
}

MODEL = LinODEnet(**MODEL_CONFIG).to(DEVICE)
MODEL = torch.jit.script(MODEL)
torchinfo.summary(MODEL, depth=2)


# %% [markdown]
# ## Warm-Up - pre-allocate memory
#
# We perform forward and backward with a maximal size batch.
#
# **Reference:** [pre-allocate-memory](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-lengthhttps://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length)

# %%
def predict_fn(model, batch) -> tuple[Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, _, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    YHAT = model(T, X)
    return Y, YHAT


batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)

# Forward
Y, YHAT = predict_fn(MODEL, batch)
assert not torch.isnan(YHAT).any(), f"prediction has NaN values!"

# Loss
R = LOSS(Y, YHAT)
assert torch.isfinite(R), "Model Collapsed!"

# Backward
R.backward()
assert not torch.isnan(MODEL.kernel.grad).any(), f"gradient has NaN values!"

# Reset
MODEL.zero_grad(set_to_none=True)

# %%
raise

# %%
predictions = YHAT
targets = Y
r = predictions - targets
m = ~torch.isnan(targets)  # 1 if not nan, 0 if nan
r = torch.where(m, r, 0.0)
r = r**2  # must come after where, else we get NaN gradients!
c = torch.sum(m, dim=-2, keepdim=True)
s = torch.sum(r / c, dim=-2, keepdim=True)
r = torch.where(c > 0, s, 0.0)
r = torch.sum(r, dim=-1, keepdim=True)
r = torch.mean(r)

# %%


# %%

# %% [markdown]
# ## Initialize Optimizer

# %%
from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)

# %% [markdown]
# ## Initialize Logging

# %%
from torch.utils.tensorboard import SummaryWriter

from tsdm.logutils import StandardLogger

WRITER = SummaryWriter(LOGGING_DIR)
LOGGER = StandardLogger(
    writer=WRITER,
    model=MODEL,
    optimizer=OPTIMIZER,
    metrics=METRICS,
    dataloaders=EVAL_LOADERS,
    hparam_dict=hparam_dict,
    checkpoint_dir=CKPOINT_DIR,
    predict_fn=predict_fn,
    results_dir=RESULTS_DIR,
)
LOGGER.log_epoch_end(0)

# %% [markdown]
# ## Training

# %%
total_num_batches = 0
for epoch in trange(1, ARGS.epochs, desc="Epoch", position=0):
    for batch in tqdm(
        TRAIN_LOADER, desc="Batch", leave=False, position=1, disable=ARGS.quiet
    ):
        total_num_batches += 1
        MODEL.zero_grad(set_to_none=True)

        # Forward
        Y, YHAT = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT)
        assert torch.isfinite(R), "Model Collapsed!"

        # Backward
        R.backward()
        OPTIMIZER.step()

        # Logging
        LOGGER.log_batch_end(total_num_batches, targets=Y, predics=YHAT)
    LOGGER.log_epoch_end(epoch)

LOGGER.log_history(CFG_ID)
LOGGER.log_hparams(CFG_ID)
