#!/usr/bin/env bash
# get project root with git
cd "$(git rev-parse --show-toplevel)" || exit
echo "Project root: $(pwd)"
python -m cProfile -o tests/import_time.prof tests/x
#python -X importtime -c "import tsdm" 2> tests/import_time.log
snakeviz tests/import_time.prof
