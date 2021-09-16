r"""Setup File.

We will move to pyproject.toml setup once PEP 660 is resolved and part of setuptools/poetry
"""

import io
import os
import re
from pathlib import Path

import pkg_resources
import setuptools

NAME = "tsdm"

with open(f"{NAME}/VERSION", "r") as file:
    VERSION = file.read()

with Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

if "CI_PIPELINE_IID" in os.environ:
    BUILD_NUMBER = os.environ["CI_PIPELINE_IID"]
    VERSION += f".dev{BUILD_NUMBER}"


def _read(filename: str) -> str:
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setuptools.setup(
    name=NAME,
    version=VERSION,
    url="https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm",
    license="MIT",
    author="Randolf Scholz",
    author_email="scholz@ismll.uni-hildesheim.de",
    description="Time Series Datasets and Models",
    long_description=_read("README.rst"),
    long_description_content_type="test/x-rst",
    packages=setuptools.find_packages(exclude=["test"]),  # include all packages in ...
    install_requires=[
        "h5py",
        "matplotlib",
        "numpy",
        "numba",
        "pandas",
        "pyyaml",
        "scipy",
        "tables",
        "torch",
        "tqdm",
        "xarray",
    ],
    # Files that listed in MANIFEST.in and also are in python packages,
    # i.e. contained in folders with and __init__.py, will be included.
    include_package_data=True,
    # ...but exclude virtualenv.yaml from all packages
    exclude_package_data={"": ["virtualenv.yaml"]},
)
