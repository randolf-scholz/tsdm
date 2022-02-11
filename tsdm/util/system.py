r"""System utilities.

Contains things like

- user queries (yes/no/choice questions)
- package installation
"""

__all__ = [
    # Functions
    "get_napoleon_type_aliases",
    "get_requirements",
    "install_package",
    "query_bool",
    "query_choice",
    "to_alphanumeric",
    "to_base",
    "write_requirements",
]

import importlib
import inspect
import logging
import string
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

from tsdm.util.strings import dict2string

__logger__ = logging.getLogger(__name__)


def get_napoleon_type_aliases(module: ModuleType) -> dict[str, str]:
    r"""Automatically create type aliases for all exported functions and classes.

    Parameters
    ----------
    module: ModuleType

    Returns
    -------
    dict[str, str]
    """
    d: dict[str, str] = {}
    for item in module.__all__:
        obj = getattr(module, item)
        if inspect.ismodule(obj):
            d[item] = f":py:mod:`~{obj.__name__}`"
            if not item.startswith("_"):
                d |= get_napoleon_type_aliases(obj)
        elif inspect.isfunction(obj):
            d[item] = f":py:func:`~{obj.__module__}.{obj.__qualname__}`"
        elif inspect.isclass(obj):
            d[item] = f":py:class:`~{obj.__module__}.{obj.__qualname__}`"
        else:
            d[item] = f":py:obj:`~{module.__name__}.{item}`"

    __logger__.info("Found napoleon type aliases: %s", dict2string(d))
    return d


def query_bool(question: str, default: Optional[bool] = True) -> bool:
    r"""Ask a yes/no question and returns answer as bool.

    Parameters
    ----------
    question: str
    default: Optional[bool], default True

    Returns
    -------
    bool
    """
    responses = {
        "y": True,
        "yes": True,
        "n": False,
        "no": False,
    }

    prompt = "([y]/n)" if default else "([n]/y)"

    while True:
        try:
            print(question)
            choice = input(prompt).lower()
        except KeyboardInterrupt as E:
            print("Operation aborted. Exiting.")
            raise E

        if not choice and default is not None:
            return default
        if choice in responses:
            return responses[choice]
        print("Please enter either of %s", responses)


def query_choice(
    question: str,
    choices: set[str],
    default: Optional[str] = None,
    pick_by_number: bool = True,
) -> str:
    r"""Ask the user to pick an option.

    Parameters
    ----------
    question: str
    choices: tuple[str]
    default: Optional[str]
    pick_by_number: bool, default True
        If True, will allow the user to pick the choice by entering its number.

    Returns
    -------
    str
    """
    choices = set(choices)
    ids: dict[int, str] = dict(enumerate(choices))

    if default is not None:
        assert default in choices

    options = "\n".join(
        f"{k}. {v}" + " (default)" * (v == default) for k, v in enumerate(choices)
    )

    while True:
        try:
            print(question)
            print(options)
            choice = input("Your choice (int or name)")
        except KeyboardInterrupt:
            print("Operation aborted. Exiting.")
            sys.exit(0)

        if choice in choices:
            return choice
        if pick_by_number and choice.isdigit() and int(choice) in ids:
            return ids[int(choice)]
        print("Please enter either of %s", choices)


# def try_import(pkgname: str):
#     """
#
#     Parameters
#     ----------
#     pkgname
#
#     Returns
#     -------
#
#     """
#     if pgkname in


def install_package(
    package_name: str,
    non_interactive: bool = False,
    installer: str = "pip",
    options: tuple[str, ...] = (),
):
    r"""Install a package via pip or other package manger.

    Parameters
    ----------
    package_name: str
    non_interactive: bool, default False
        If false, will generate a user prompt.
    installer: str, default "pip"
        Can also use `conda` or `mamba`
    options: tuple[str, ...]
        Options to pass to the isntaller
    """
    package_available = importlib.util.find_spec(package_name)
    install_call = (installer, "install", package_name)
    if not package_available:
        if non_interactive or query_bool(
            f"Package '{package_name}' not found. Do you want to install it?"
        ):
            try:
                subprocess.run(install_call + options, check=True)
            except subprocess.CalledProcessError as E:
                raise RuntimeError("Execution failed with error") from E
    else:
        __logger__.info("Package '%s' already installed.", package_name)


def get_requirements(package: str, version: Optional[str] = None) -> dict[str, str]:
    r"""Return dictionary containing requirements with version numbers.

    Parameters
    ----------
    package: str
    version: Optional[str]
        In the case of None, the latest version is used.

    Returns
    -------
    dict[str, str]
    """
    # get requirements as string of the form package==version\n.
    reqs = subprocess.check_output(
        (
            r"johnnydep",
            f"{package}" + f"=={version}" * bool(version),
            r"--output-format",
            r"pinned",
        ),
        text=True,
    )
    return dict(line.split("==") for line in reqs.rstrip("\n").split("\n"))


def write_requirements(
    package: str, version: Optional[str] = None, path: Optional[Path] = None
):
    r"""Write a requirements dictionary to a requirements.txt file.

    Parameters
    ----------
    package: str
    version: Optional[str]
        In the case of ``None``, the latest version is used.
    path: Optional[Path]
        In the case of ``None``, "requirements" is used.
    """
    requirements: dict[str, str] = get_requirements(package, version)
    # Note: the first entry is the package itself!
    fname = f"requirements-{package}=={requirements.pop(package)}.txt"
    path = Path("requirements") if path is None else Path(path)
    with open(path.joinpath(fname), "w", encoding="utf8") as file:
        file.write("\n".join(f"{k}=={requirements[k]}" for k in sorted(requirements)))


def to_base(n: int, b: int) -> list[int]:
    r"""Convert non-negative integer to any basis.

    References
    ----------
    - https://stackoverflow.com/a/28666223/9318372

    Parameters
    ----------
    n: int
    b: int

    Returns
    -------
    digits: list[int]
        Satisfies: ``n = sum(d*b**k for k, d in enumerate(reversed(digits)))``
    """
    digits = []
    while n:
        n, d = divmod(n, b)
        digits.append(d)
    return digits[::-1] or [0]


def to_alphanumeric(n: int) -> str:
    r"""Convert integer to alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    digits = to_base(n, len(chars))
    return "".join(chars[i] for i in digits)


# def shorthash(inputs) -> str:
#     r"""Roughly good for 2¹⁶=65536 items."""
#     encoded = json.dumps(dictionary, sort_keys=True).encode()
#
#     return shake_256(inputs).hexdigest(8)

# from typing import Union
# from datetime import datetime, timedelta
#
# BaseTypes = Union[None, bool, int, float, str, datetime, timedelta]
# ListType = list[BaseTypes]
# NestedListType = Union[]
# RecursiveListType = Union[ListType, list[ListType], list[list[ListType]]]
# DictType = dict[str, BaseTypes]
# ContainerTypes = Union[list[ListType], list[DictType], dict[str, ListType], dict[str, DictType]]
# AllowedTypes = Union[BaseTypes, ListType, DictType, ContainerTypes]
# NestedType1 = dict[str, AllowedTypes]
# NestedType2 = dict[str, Union[AllowedTypes, ]
#
#
# dict[str, AllowedTypes, NestedType]
# JSON = dict[str, "JSON"]
