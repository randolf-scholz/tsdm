#!/usr/bin/env python
"""Updates the pyproject.toml dependencies to the currently installed versions."""

import json
import re
import subprocess

# run pip list with json format and load the output into a dictionary
pip_list = json.loads(subprocess.check_output(["pip", "list", "--format=json"]))
pkg_dict = {pkg["name"].lower(): pkg["version"] for pkg in pip_list}


def strip(version: str) -> str:
    """Strip the version string to the first three parts."""
    sub = version.split(".")
    version = ".".join(sub[:3])
    # strip everything after the first non-numeric, non-dot character
    version = re.sub(r"[^0-9\.].*", "", version)
    return version


# regex to match the dependencies in pyproject.toml
regex = r'([a-zA-Z0-9_-]*) = ">=([0-9.]*)"'
regex = re.compile(regex)

with open("pyproject.toml", "r", encoding="utf8") as file:
    pyproject = file.read()

matches = regex.findall(pyproject)
for match in matches:
    pkg, old_version = match
    new_version = strip(pkg_dict.get(pkg, old_version))
    if old_version != new_version:
        old = f'{pkg} = ">={old_version}"'
        new = f'{pkg} = ">={new_version}"'
        print(f"replacing: {old!r:36}  {new!r}")
        pyproject = pyproject.replace(old, new)

with open("pyproject.toml", "w", encoding="utf8") as file:
    file.write(pyproject)
