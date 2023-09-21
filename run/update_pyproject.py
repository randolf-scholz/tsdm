#!/usr/bin/env python
"""Updates the pyproject.toml dependencies to the currently installed versions."""

import json
import re
import subprocess
from typing import Literal

# run pip list with json format and load the output into a dictionary
pip_list: list[dict[Literal["name", "version"], str]] = json.loads(
    subprocess.check_output(["pip", "list", "--format=json"])
)
pkg_dict: dict[str, str] = {pkg["name"].lower(): pkg["version"] for pkg in pip_list}


def strip(version: str) -> str:
    """Strip the version string to the first three parts."""
    sub = version.split(".")
    version = ".".join(sub[:3])
    # strip everything after the first non-numeric, non-dot character
    version = re.sub(r"[^0-9\.].*", "", version)
    return version


def update_versions(raw_file: str, version_pattern: re.Pattern) -> str:
    """Update the dependencies in pyproject.toml according to version_pattern."""
    if version_pattern.groups != 3:
        raise ValueError(
            "version_pattern must have 3 groups (whole match, package name, version))"
        )

    # match all dependencies in the file
    for match, pkg, old_version in version_pattern.findall(raw_file):
        # get the new version from the pip list
        new_version = strip(pkg_dict.get(pkg, old_version))
        # if the version changed, replace the old version with the new one
        if old_version != new_version:
            new = match.replace(old_version, new_version)
            print(f"replacing: {match!r:36}  {new!r}")
            raw_file = raw_file.replace(match, new)
    return raw_file


with open("pyproject.toml", "r", encoding="utf8") as file:
    pyproject = file.read()

# update pyproject.dependencies
pyproject_pattern = re.compile(r'"(([a-zA-Z0-9_-]*)>=([0-9.]*))')
pyproject = update_versions(pyproject, pyproject_pattern)

# update tool.poetry.dependencies
poetry_pattern = re.compile(r'(([a-zA-Z0-9_-]*) = ">=([0-9.]*))')
pyproject = update_versions(pyproject, poetry_pattern)

# needed for things like `black = {version = ">=23.7.0", extras = ["d", "jupyter"]}`
version_pattern = re.compile(r'(([a-zA-Z0-9_-]*) = \{\s?version = ">=([0-9.]*))')
pyproject = update_versions(pyproject, version_pattern)

with open("pyproject.toml", "w", encoding="utf8") as file:
    # update the file
    file.write(pyproject)
