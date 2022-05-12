#!/usr/bin/env python

import os
import subprocess
import sys

MAX_KEY_LEN = max(len(key) for key in os.environ)


def header(s: str, pad=3):
    n = (79 - len(s)) // 2 - pad
    s = s.upper().center(79 - 2 * n)
    print(f"\n{'>'*n + s + '<'*n}\n", file=sys.stdout, flush=True)


def subheader(s: str):
    print(f"\n{s.upper()}\n{'-' * len(s)}", flush=True)


def pprint_dict(d: dict):
    for key, val in d.items():
        print(f"{key.ljust(MAX_KEY_LEN)} : {val}")
    sys.stdout.flush()


###############################################################################
header("DEBUGGING STATS")  #
############################

SLURM_JOBID = os.environ.get("SLURM_JOBID")
STDOUT_LOG = subprocess.run(
    f"scontrol show jobid={SLURM_JOBID}" + r"| grep -Po 'StdOut=\K.*$' | tr -d '\n'",
    shell=True,
    encoding="utf-8",
    capture_output=True,
).stdout
STDERR_LOG = subprocess.run(
    f"scontrol show jobid={SLURM_JOBID}" + r"| grep -Po 'StdErr=\K.*$' | tr -d '\n'",
    shell=True,
    encoding="utf-8",
    capture_output=True,
).stdout

debug_stats = {
    "USER": os.environ.get("USER"),
    "USERNAME": os.environ.get("USERNAME"),
    "Virtual Environment": sys.prefix,
    "Python Script": os.getcwd() + "/" + sys.argv[0],
    "CPU Count": os.cpu_count(),
    "CPU Allocation": os.sched_getaffinity(0),
    "PATH": os.environ.get("PATH").split(":"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH").split(":"),
    "STDOUT_LOG": STDOUT_LOG,
    "STDERR_LOG": STDERR_LOG,
    "SLURM_JOBID": SLURM_JOBID,
    "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME"),
}
pprint_dict(debug_stats)

###############################################################################
header("ENVIRONMENT VARIABLES")  #
##################################

for word in "conda", "cuda", "mkl", "tf", "xla", "slurm", "omp", "kmp":
    subheader(f"{word} configuration")
    matches = {}
    for key in sorted(os.environ):
        if f"{word}_" in key.lower() or f"_{word}" in key.lower():
            matches[key] = os.environ.get(key)
    pprint_dict(matches)


###############################################################################
header("user soft limits")  #
#############################

print(
    subprocess.run(
        f"ulimit -Sa",
        shell=True,
        encoding="utf-8",
        capture_output=True,
    ).stdout
)

###############################################################################
header("user hard limits")  #
#############################

print(
    subprocess.run(
        f"ulimit -Ha",
        shell=True,
        encoding="utf-8",
        capture_output=True,
    ).stdout
)

###############################################################################
header("security limits")  #
############################

print(
    subprocess.run(
        f"cat /etc/security/limits.conf",
        shell=True,
        encoding="utf-8",
        capture_output=True,
    ).stdout
)

###############################################################################
header("shared memory limits")  #
#################################

print(
    subprocess.run(
        f"ipcs -lm",
        shell=True,
        encoding="utf-8",
        capture_output=True,
    ).stdout
)


###############################################################################
header("SLURM STATUS")  #
#########################

print(
    subprocess.run(
        f"scontrol show -d jobid {os.environ.get('SLURM_JOB_ID')}",
        shell=True,
        encoding="utf-8",
        capture_output=True,
    ).stdout
)

###############################################################################
header("CONDA PACKAGES")  #
###########################

print(
    subprocess.run(
        "conda list", shell=True, encoding="utf-8", capture_output=True
    ).stdout
)

###############################################################################
header("NVIDIA STATS")  #
#########################

print(
    subprocess.run(
        "ptxas --version", shell=True, encoding="utf-8", capture_output=True
    ).stdout
)
print(
    subprocess.run(
        "nvcc --version", shell=True, encoding="utf-8", capture_output=True
    ).stdout
)
print(
    subprocess.run(
        "nvidia-smi", shell=True, encoding="utf-8", capture_output=True
    ).stdout
)
sys.stdout.flush()
