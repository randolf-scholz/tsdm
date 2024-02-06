#!/usr/bin/env bash
set -e  # exit on first error

# navigate to the root of the project
cd "$(git rev-parse --show-toplevel)"
echo "$PWD"

# update dependencies
pre-commit autoupdate
pdm update --update-all

# commit the changes
set +e
pre-commit run pyproject-update-deps -a --hook-stage manual
git add .
git commit -m "chore(deps): update dependencies"
