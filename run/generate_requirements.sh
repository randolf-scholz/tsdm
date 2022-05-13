#!/usr/bin/env bash
poetry export -f requirements.txt --output requirements.txt --extras all
poetry export -f requirements.txt --output requirements-dev.txt --dev --extras all
