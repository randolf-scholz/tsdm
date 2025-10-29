#!/usr/bin/env python

import argparse
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def run_mypy_timed(path: Path, /, *, show_errors: bool = False):
    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["mypy", "--no-incremental", "--cache-dir=/dev/null", str(path)],
            stdout=None if show_errors else subprocess.DEVNULL,
            stderr=None if show_errors else subprocess.DEVNULL,
            check=False,
        )
    except Exception as e:
        print(f"Error running mypy on {path}: {e}")
    end = time.perf_counter()
    return end - start


def analyze_project(root_dir: Path, *, limit: int, show_errors: bool):
    files = sorted(root_dir.rglob("*.py"))
    timings = []

    print(f"Running mypy on {len(files)} files...\n")
    for file in (pbar := tqdm(files, desc="Checking")):
        pbar.set_description(f"Processing {file}")
        duration = run_mypy_timed(file, show_errors)
        timings.append((duration, file))

    timings.sort(reverse=True)
    print(f"\nTop {limit} slowest files (with incremental mode disabled):\n")
    for duration, file in timings[:limit]:
        print(f"{duration:.2f}s  {file}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure mypy execution time per Python file in a project (no incremental mode)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the root of the project or source tree",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of slowest files to display (default: 30)",
    )
    parser.add_argument(
        "--show-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        type=bool,
        help="Print mypy output instead of suppressing it",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: path {args.path} does not exist.")
        raise SystemExit(1)

    analyze_project(args.path, args.limit, args.show_errors)


if __name__ == "__main__":
    main()
