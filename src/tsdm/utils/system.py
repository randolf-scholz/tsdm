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
    "write_requirements",
]

import importlib
import inspect
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

from tsdm.utils.strings import repr_mapping

__logger__ = logging.getLogger(__name__)


def get_requirements(
    package: str, /, *, version: Optional[str] = None
) -> dict[str, str]:
    r"""Return dictionary containing requirements with version numbers.

    If `version=None`, then the latest version is used.
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


def get_napoleon_type_aliases(module: ModuleType) -> dict[str, str]:
    r"""Automatically create type aliases for all exported functions and classes."""
    d: dict[str, str] = {}
    if not hasattr(module, "__all__"):
        return d

    # for item in module.__all__:
    #     obj = getattr(module, item)
    #     if inspect.ismodule(obj):
    #         # d[item] = f":mod:`~{obj.__name__}`"
    #         if not item.startswith("_"):
    #             d |= get_napoleon_type_aliases(obj)
    #     if hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
    #         d[item] = f"{obj.__module__}.{obj.__qualname__}"
    # elif inspect.ismethod(obj):
    #     d[item] = f":meth:`~{obj.__module__}.{obj.__qualname__}`"
    # elif inspect.isfunction(obj):
    #     d[item] = f":func:`~{obj.__module__}.{obj.__qualname__}`"
    # elif inspect.isclass(obj):
    #     if issubclass(obj, Exception):
    #         d[item] = f":exc:`~{obj.__module__}.{obj.__qualname__}`"
    #     d[item] = f":class:`~{obj.__module__}.{obj.__qualname__}`"
    # else:
    #     pass

    # d[item] = f":obj:`~{module.__name__}.{item}`"

    for item in module.__all__:
        obj = getattr(module, item)
        if inspect.ismodule(obj):
            d[item] = f"{obj.__name__}"
            if not item.startswith("_"):
                d |= get_napoleon_type_aliases(obj)
        elif inspect.ismethod(obj):
            d[item] = f"{obj.__module__}.{obj.__qualname__}"
        elif inspect.isfunction(obj):
            d[item] = f"{obj.__module__}.{obj.__qualname__}"
        elif inspect.isclass(obj):
            if issubclass(obj, Exception):
                d[item] = f"{obj.__module__}.{obj.__qualname__}"
            d[item] = f"{obj.__module__}.{obj.__qualname__}"
        else:
            d[item] = item

    # for item in module.__all__:
    #     obj = getattr(module, item)
    #     if inspect.ismodule(obj):
    #         d[item] = f":mod:`~{obj.__name__}`"
    #         if not item.startswith("_"):
    #             d |= get_napoleon_type_aliases(obj)
    #     elif inspect.ismethod(obj):
    #         d[item] = f":meth:`~{obj.__module__}.{obj.__qualname__}`"
    #     elif inspect.isfunction(obj):
    #         d[item] = f":func:`~{obj.__module__}.{obj.__qualname__}`"
    #     elif inspect.isclass(obj):
    #         if issubclass(obj, Exception):
    #             d[item] = f":exc:`~{obj.__module__}.{obj.__qualname__}`"
    #         d[item] = f":class:`~{obj.__module__}.{obj.__qualname__}`"
    #     else:
    #         d[item] = f":obj:`~{module.__name__}.{item}`"

    __logger__.info("Found napoleon type aliases: %s", repr_mapping(d, maxitems=-1))
    return d


def query_bool(question: str, /, *, default: Optional[bool] = True) -> bool:
    r"""Ask a yes/no question and returns answer as bool."""
    responses = {"y": True, "yes": True, "n": False, "no": False}
    prompt = "([y]/n)" if default else "([n]/y)"

    while True:
        try:
            print(question)
            choice = input(prompt).lower()
        except KeyboardInterrupt:
            # TODO: use exception notes
            print("Operation aborted.")
            sys.exit(0)

        if not choice and default is not None:
            return default
        if choice in responses:
            return responses[choice]
        print("Please enter either of %s", responses)


def query_choice(
    question: str,
    /,
    *,
    choices: set[str],
    default: Optional[str] = None,
    pick_by_number: bool = True,
) -> str:
    r"""Ask the user to pick an option.

    If `pick_by_number=True`, then will allow the user to pick the choice by number.
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
            # TODO: use exception notes
            print("Operation aborted.")
            sys.exit(0)

        if choice in choices:
            return choice
        if pick_by_number and choice.isdigit() and int(choice) in ids:
            return ids[int(choice)]
        print("Please enter either of %s", choices)


def install_package(
    package_name: str,
    /,
    *,
    non_interactive: bool = False,
    installer: str = "pip",
    options: tuple[str, ...] = (),
) -> None:
    r"""Install a package via pip or other package manager.

    Args:
        package_name: str
        non_interactive: If False, will generate a user prompt.
        installer: Can also use `conda` or `mamba`
        options: Options to pass to the installer
    """
    package_available = importlib.util.find_spec(package_name)
    install_call = (installer, "install", package_name)
    if not package_available:
        if non_interactive or query_bool(
            f"Package {package_name!r} not found. Do you want to install it?"
        ):
            try:
                subprocess.run(install_call + options, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Execution failed with error") from exc
    else:
        __logger__.info("Package '%s' already installed.", package_name)


def write_requirements(
    path: Optional[Path] = None,
    /,
    *,
    package: str,
    version: Optional[str] = None,
) -> None:
    r"""Write a requirements dictionary to a requirements.txt file.

    If `version=None`, then the latest version is used.
    """
    requirements: dict[str, str] = get_requirements(package, version=version)
    # Note: the first entry is the package itself!
    fname = f"requirements-{package}=={requirements.pop(package)}.txt"
    path = Path("requirements") if path is None else Path(path)
    with open(path.joinpath(fname), "w", encoding="utf8") as file:
        file.write("\n".join(f"{k}=={requirements[k]}" for k in sorted(requirements)))
