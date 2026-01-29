r"""Utilities for models."""

__all__ = [
    "autojit",
    "initialize_from_config",
    "import_module_from_path",
    "install_package",
    "get_requirements",
    "write_requirements",
]

import logging
import subprocess
from functools import wraps
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Self

from torch import jit, nn

from tsdm.config import CONFIG
from tsdm.types.aliases import DirPath
from tsdm.utils._utils import query_bool
from tsdm.utils.contextmanagers import system_path


def autojit[M: nn.Module](base_class: type[M], /) -> type[M]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule: ...


        model = jit.script(MyModule())

    and

    .. code-block:: python

        @autojit
        class MyModule: ...


        model = MyModule()

    are (roughly?) equivalent
    """
    if not isinstance(base_class, type):
        raise TypeError("Expected a class.")
    if not issubclass(base_class, nn.Module):
        raise TypeError("Expected a subclass of nn.Module.")

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[valid-type,misc]
        r"""A simple Wrapper."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            # Note: If __new__() does not return an instance of cls,
            #   then the new instance's __init__() method will not be invoked.
            instance = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted = jit.script(instance)
                return scripted  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
            return instance  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    if not isinstance(WrappedClass, type):
        raise TypeError(f"Expected a class, got {WrappedClass}.")
    if not issubclass(WrappedClass, base_class):
        raise TypeError(f"Expected {WrappedClass} to be a subclass of {base_class}.")

    return WrappedClass  # pyright: ignore[reportReturnType]


def initialize_from_config(config: dict[str, Any], /) -> nn.Module:
    r"""Initialize `nn.Module` from a config object."""
    conf = config.copy()
    cls_name: str = conf.pop("__name__")
    module_name: str = conf.pop("__module__")

    # drop other dunder keys
    opts = {k: v for k, v in conf.items() if not k.startswith("__")}

    # import module and class
    module = import_module_from_path(module_name)
    cls = getattr(module, cls_name)

    # initialize class with options
    try:
        obj = cls(**opts)
    except Exception as exc:
        exc.add_note(f"Failed to initialize {cls_name} with {opts}.")
        raise

    return obj


def import_module_from_path(
    module_dir: DirPath, /, *, module_name: Optional[str] = None
) -> ModuleType:
    r"""Return python module imported from the path.

    References:
        - https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        - https://stackoverflow.com/a/41904558
    """
    module_path = Path(module_dir)
    module_init = module_path / "__init__.py"
    module_name = module_name or module_path.parts[-1]

    # validate that the module has an __init__ file.
    if not module_init.exists():
        raise FileNotFoundError(f"Module {module_path} has no __init__ file !")

    with system_path(module_path):
        spec = spec_from_file_location(module_name, str(module_init))
        assert spec is not None
        assert spec.loader is not None
        the_module = module_from_spec(spec)
        spec.loader.exec_module(the_module)
        return the_module


def get_requirements(
    package_name: str, /, *, version: Optional[str] = None
) -> dict[str, str]:
    r"""Return dictionary containing requirements with version numbers.

    If `version=None`, then the latest version is used.
    """
    # get requirements as string of the form package==version\n.
    reqs = subprocess.check_output(
        (
            r"johnnydep",
            f" {package_name}" + f"=={version}" * bool(version),
            r" --output-format",
            r" pinned",
        ),
        text=True,
    )
    return dict(line.split("==") for line in reqs.rstrip("\n").split("\n"))


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
    file = path / fname
    text = "\n".join(f"{k}=={requirements[k]}" for k in sorted(requirements))
    file.write_text(text, encoding="utf8")


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
            f"Package {package_name!r} not found. Do you want to install it?",
            default=True,
        ):
            try:
                subprocess.run(install_call + options, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Execution failed with error") from exc
    else:
        logger = logging.getLogger(f"{__name__}/{install_package.__name__}")
        logger.info("Package '%s' already installed.", package_name)
