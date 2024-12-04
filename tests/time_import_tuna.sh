#!/usr/bin/env bash
# get project root with git
cd "$(git rev-parse --show-toplevel)" || exit
echo "Project root: $(pwd)"
python -X importtime -c "import tsdm" 2> tests/import_time.log
tuna tests/import_time.log
