#!/usr/bin/env python
"""Autogenerate SBATCH files for Grid search."""
import argparse
import os
import socket
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml

NOW = datetime.now().isoformat(timespec="seconds")

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("-r",  "--run_id",    default=None,   type=str,   help="run_id")
parser.add_argument("-d",  "--dataset",   default=None,   type=str,   help="dataset name")
parser.add_argument("-c",  "--config",    default=None,   type=str,   help="config file")
parser.add_argument("-m",  "--model",     default=None,   type=str,   help="model name")
parser.add_argument("-s",  "--script",    default=None,   type=str,   help="script name")
parser.add_argument("-p",  "--partition", default=None,   type=str,   help="partition name")
# fmt: on

ARGS = parser.parse_args()
MODEL = ARGS.model or "LinODEnet"
DATASET = ARGS.dataset or "KIWI"
PARTITION = ARGS.partition or "NGPU"
SCRIPT = ARGS.script or f"script-{DATASET.lower()}.py"
RUN_ID = ARGS.run_id or NOW
CFG_ID = ARGS.config or f"{DATASET}-{MODEL}-{RUN_ID}.yaml"


HOME = Path.cwd()

CONFIG_DIR = HOME / "configs"
CONFIG_DIR.mkdir(exist_ok=True, parents=True)

LOG_DIR = HOME / "logs" / DATASET / RUN_ID
LOG_DIR.mkdir(exist_ok=True, parents=True)

SLURM_DIR = HOME / "slurm" / DATASET / RUN_ID
SLURM_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = HOME / "results" / DATASET / MODEL / RUN_ID

USER = os.environ.get("USER")
DOMAIN = socket.getfqdn().split(".", 1)[1]

CONFIG_FILE = CONFIG_DIR / CFG_ID

CFG = {
    "fold": [0, 1, 2, 3, 4],
    "epochs": [100],
    "batch_size": [64],
    "learn_rate": [0.001],
    "hidden_size": [64, 128],
    "latent_size": [128, 192],
    "seed": [None],
}

# it's black magic
GRID = dict(enumerate(dict(zip(CFG.keys(), hpc)) for hpc in product(*CFG.values())))

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as file:
        GRID = yaml.safe_load(file)
else:
    with open(CONFIG_FILE, "w") as file:
        yaml.safe_dump(GRID, file)

SBATCH = "\n".join(
    [
        r"#!/usr/bin/env bash",
        f"#SBATCH --job-name={DATASET}",
        f"#SBATCH --partition={PARTITION}",
        f"#SBATCH --output={LOG_DIR / r'%j.stdout.log'}",
        f"#SBATCH --error={LOG_DIR / r'%j.stderr.log'}",
        r"#SBATCH --mail-type=FAIL",
        f"#SBATCH --mail-user={USER}@{DOMAIN}",
        r"#SBATCH --gpus=1",
        r"",
        r"mkdir -p logs",
        r"ulimit -Sn 32768",
        r"source activate kiwi",
        f"srun python {SCRIPT} -q -r {RUN_ID} --config {CONFIG_FILE}",
    ]
)

print("\n".join([f"{key}={val}" for key, val in vars().items() if key.isupper()]))

for ID in GRID:
    # Write and Execute SBATCH
    sbatch_file = SLURM_DIR / f"sbatch-{ID}.sh"
    with open(sbatch_file, "w") as file:
        file.write(SBATCH + f" {ID}\n")

    fname = RESULTS_DIR / f"{ID}.yaml"

    if fname.is_file():
        continue  # skip!

    os.system(f"sbatch {sbatch_file}")
