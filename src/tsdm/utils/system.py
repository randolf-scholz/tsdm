r"""System utilities.

Contains things like

- user queries (yes/no/choice questions)
- package installation
"""

__all__ = [
    # Functions
    "get_napoleon_type_aliases",
    "get_requirements",
    "import_module",
    "install_package",
    "query_bool",
    "query_choice",
    "write_requirements",
]

import inspect
import logging
import subprocess
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from typing_extensions import Optional

from tsdm.utils.contextmanagers import add_to_path
from tsdm.utils.pprint import repr_mapping

__logger__: logging.Logger = logging.getLogger(__name__)


def import_module(
    module_path: Path, /, *, module_name: Optional[str] = None
) -> ModuleType:
    r"""Return python module imported from the path.

    References:
        - https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        - https://stackoverflow.com/a/41904558
    """
    module_name = module_name or module_path.parts[-1]
    module_init = module_path.joinpath("__init__.py")
    assert module_init.exists(), f"Module {module_path} has no __init__ file !!!"

    with add_to_path(module_path):
        spec = spec_from_file_location(module_name, str(module_init))
        the_module = module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(the_module)  # type: ignore[union-attr]
        return the_module


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
            f" {package}" + f"=={version}" * bool(version),
            r" --output-format",
            r" pinned",
        ),
        text=True,
    )
    return dict(line.split("==") for line in reqs.rstrip("\n").split("\n"))


def get_napoleon_type_aliases(module: ModuleType) -> dict[str, str]:
    r"""Automatically create type aliases for all exported functions and classes."""
    d: dict[str, str] = {}
    if not hasattr(module, "__all__"):
        return d

    for item in module.__all__:
        obj = getattr(module, item)
        if inspect.ismodule(obj):
            d[item] = f"{obj.__name__}"
            if not item.startswith("_"):
                d |= get_napoleon_type_aliases(obj)
        elif inspect.ismethod(obj) or inspect.isfunction(obj):
            d[item] = f"{obj.__module__}.{obj.__qualname__}"
        elif inspect.isclass(obj):
            if issubclass(obj, Exception):
                d[item] = f"{obj.__module__}.{obj.__qualname__}"
            d[item] = f"{obj.__module__}.{obj.__qualname__}"
        else:
            d[item] = item

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
        except KeyboardInterrupt as exc:
            exc.add_note("Operation aborted.")
            raise

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
        except KeyboardInterrupt as exc:
            exc.add_note("Operation aborted.")
            raise

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
    package_available = find_spec(package_name)
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
    r"""Write a 'requirements'-dictionary to a `requirements.txt` file.

    If `version=None`, then the latest version is used.
    """
    requirements: dict[str, str] = get_requirements(package, version=version)
    # Note: the first entry is the package itself!
    fname = f"requirements-{package}=={requirements.pop(package)}.txt"
    path = Path("requirements") if path is None else Path(path)
    file = path.joinpath(fname)
    text = "\n".join(f"{k}=={requirements[k]}" for k in sorted(requirements))
    file.write_text(text, encoding="utf8")
