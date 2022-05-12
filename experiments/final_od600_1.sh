#!/usr/bin/env bash
#SBATCH --job-name=FINAL_OD600
#SBATCH --partition=GPU
#SBATCH --output=logs/%j-%x.stdout.log
#SBATCH --error=logs/%j-%x.stderr.log
#SBATCH --gpus=1
#SBATCH --exclude=pgpu-[020-021],ngpu-022

source activate kiwi
ulimit -Sn 32768
mkdir -p logs

srun python debug_info.py

srun python KIWI_FINAL_PRODUCT.py "OD600" --split=1
