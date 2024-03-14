#!/usr/bin/env bash
# execute pydeps and create dependency graph
# get the project name
project_name=$(basename "$(pwd)")
pydeps "src/$project_name" --cluster --rankdir BT --max-bacon=1
