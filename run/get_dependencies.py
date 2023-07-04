#!/usr/bin/env python
"""Prints the direct dependencies of a module line by line.."""

import ast
import importlib
import pkgutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def extract_imports(node: ast.Import) -> set[str]:
    """Extract third-party dependencies from an `import` statement node."""
    dependencies = set()
    for alias in node.names:
        module = alias.name.split(".")[0]
        if not module.startswith("_"):
            dependencies.add(module)
    return dependencies


def extract_import_from(node: ast.ImportFrom) -> set[str]:
    """Extract third-party dependencies from an `import from` statement node."""
    dependencies = set()
    module = node.module.split(".")[0]
    if not module.startswith("_"):
        dependencies.add(module)
    return dependencies


def is_submodule(submodule_name: str, module_name: str) -> bool:
    """True if submodule_name is a submodule of module_name."""
    return submodule_name.startswith(module_name + ".")


def get_file_dependencies(file_path: Path) -> set[str]:
    """Retrieve the list of third-party dependencies imported by a file."""
    dependencies = set()
    path = Path(file_path)

    if path.suffix != ".py":
        return dependencies

    with open(path, "r", encoding="utf8") as file:
        tree = ast.parse(file.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies |= extract_imports(node)
            elif isinstance(node, ast.ImportFrom):
                dependencies |= extract_import_from(node)
    return dependencies


def get_module_dependencies(module_name: str, recursive: bool = True) -> set[str]:
    """Retrieve the list of third-party dependencies imported by a module."""
    module = importlib.import_module(module_name)
    dependencies = set()

    # Visit the current module
    with open(module.__file__, "r", encoding="utf8") as file:
        tree = ast.parse(file.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies |= extract_imports(node)
            elif isinstance(node, ast.ImportFrom):
                dependencies |= extract_import_from(node)

    if not recursive:
        return dependencies

    # if it is a package, recurse into it.
    if hasattr(module, "__path__"):
        # Visit the sub-packages/modules of the package
        for module_info in pkgutil.walk_packages(module.__path__):
            submodule = importlib.import_module(module_name + "." + module_info.name)
            submodule_name = submodule.__name__

            if is_submodule(submodule_name, module_name):
                dependencies |= get_module_dependencies(
                    submodule_name, recursive=recursive
                )

    return dependencies


def group_dependencies(dependencies: set[str]) -> tuple[list[str], list[str]]:
    """Splits the dependencies into first-party and third-party."""
    stdlib_dependencies = set()
    third_party_dependencies = set()

    for dependency in dependencies:
        if dependency in sys.stdlib_module_names:
            stdlib_dependencies.add(dependency)
        else:
            third_party_dependencies.add(dependency)

    return sorted(stdlib_dependencies), sorted(third_party_dependencies)


def collect_dependencies(name: str | Path) -> set[str]:
    """Collect the third-party dependencies from files in the given path."""
    dependencies = set()
    path = Path(name)

    if not path.exists():  # assume module
        dependencies = get_module_dependencies(name)
    elif path.is_file():  # Single file
        dependencies |= get_file_dependencies(path)
    elif path.is_dir():  # Directory
        for file_path in path.rglob("*"):
            if file_path.is_file():
                dependencies |= get_file_dependencies(file_path)
    else:
        raise ValueError(f"Invalid path: {path}")

    return dependencies


def main() -> None:
    """Print the third-party dependencies of a module."""
    if len(sys.argv) != 2:
        print("Usage: python get_dependencies.py [module_name]")
        sys.exit(1)

    name = sys.argv[1]

    with redirect_stdout(None), redirect_stderr(None):
        dependencies = collect_dependencies(name)

    _, third_party_dependencies = group_dependencies(dependencies)

    for dependency in third_party_dependencies:
        print(dependency)


if __name__ == "__main__":
    main()
