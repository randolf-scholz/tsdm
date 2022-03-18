#!/usr/bin/env bash
#SBATCH --job-name=KIWI
#SBATCH --partition=PGPU
#SBATCH --output=logs/%j-%x.stdout.log
#SBATCH --error=logs/%j-%x.stderr.log
#SBATCH --gpus=1


source activate kiwi
mkdir -p logs
srun python KIWI_RUNS_EXPERIMENT_iresnet.py
