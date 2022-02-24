#!/usr/bin/env python
# coding: utf-8


import textwrap
import subprocess
from itertools import product

for target, split in product(["OD600", "Fluo_GFP"], [0, 1, 2, 3, 4]):

    s = f"""\
    #!/usr/bin/env bash
    #SBATCH --job-name={target}
    #SBATCH --partition=GPU
    #SBATCH --output=logs/%j-%x.stdout.log
    #SBATCH --error=logs/%j-%x.stderr.log
    #SBATCH --gpus=1
    #SBATCH --exclude=pgpu-[020-021],ngpu-022

    source activate kiwi
    ulimit -Sn 32768
    mkdir -p logs

    srun python debug_info.py

    srun python KIWI_FINAL_PRODUCT.py  "{target}" --split={split}\
    """

    with open("sbatch.sh", "w") as f:
        f.write(textwrap.dedent(s))

    print(target, split)
    job = subprocess.run(
        "sbatch sbatch.sh",
        shell=True,
        encoding="utf-8",
        capture_output=True,
    )
    print(job.stdout, job.stderr)
