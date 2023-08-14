#!/usr/bin/env python
"""Prints the direct dependencies of a module line by line.."""

# TODO: add support for extras.

__all__ = [
    "get_deps_file",
    "get_deps_module",
    "group_dependencies",
    "collect_dependencies",
    "main",
    "get_deps_import",
    "get_deps_importfrom",
]

import argparse
import ast
import importlib
import pkgutil
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

import tomllib


def get_deps_import(node: ast.Import, /) -> set[str]:
    """Extract dependencies from an `import ...` statement."""
    dependencies = set()
    for alias in node.names:
        module = alias.name.split(".")[0]
        if not module.startswith("_"):
            dependencies.add(module)
    return dependencies


def get_deps_importfrom(node: ast.ImportFrom, /) -> set[str]:
    """Extract dependencies from an `from y import ...` statement."""
    dependencies = set()
    assert node.module is not None
    module = node.module.split(".")[0]
    if not module.startswith("_"):  # ignore _private modules
        dependencies.add(module)
    return dependencies


def get_deps_tree(tree: ast.AST, /) -> set[str]:
    """Extract the set of dependencies from `ast.AST` object."""
    dependencies: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dependencies |= get_deps_import(node)
        elif isinstance(node, ast.ImportFrom):
            dependencies |= get_deps_importfrom(node)

    return dependencies


def get_deps_file(file_path: str | Path, /) -> set[str]:
    """Extract set of dependencies imported by a script."""
    path = Path(file_path)

    if path.suffix != ".py":
        raise ValueError(f"Invalid file extension: {path.suffix}")

    with open(path, "r", encoding="utf8") as file:
        tree = ast.parse(file.read())

    return get_deps_tree(tree)


def get_deps_module(module: str | ModuleType, /, *, silent: bool = True) -> set[str]:
    """Extract set of dependencies imported by a module."""
    # NOTE: Generally there is no correct way to do it without importing the module.
    # This is because modules can be imported dynamically.

    if isinstance(module, str):
        with (  # load the submodule silently
            redirect_stdout(None if silent else sys.stdout),
            redirect_stderr(None if silent else sys.stderr),
        ):
            module = importlib.import_module(module)

    # Visit the current module
    dependencies = get_deps_file(module.__file__)

    if not hasattr(module, "__path__"):
        # note: can only recurse into packages.
        return dependencies

    # Visit the sub-packages/modules of the package
    # TODO: add dynamically imported submodules using the `pkgutil` module.
    for module_info in pkgutil.walk_packages(module.__path__):
        submodule_name = f"{module.__name__}.{module_info.name}"
        dependencies |= get_deps_module(submodule_name, silent=silent)

    return dependencies


def get_deps_pyproject_section(pyproject: dict, section: str, /) -> set[str]:
    """Get the dependencies from a section of pyproject.toml.

    Looking up the section must either result in a list of strings or a dict.
    """
    keys = section.split(".")

    # recursively get the section
    try:
        for key in keys:
            pyproject = pyproject[key]
    except KeyError:
        deps = NotImplemented
    else:
        match pyproject:
            case list():  # assume format `"pacakge<comparator>version"`
                regex = re.compile(r"[a-zA-Z0-9_-]*")  # package name
                deps = {re.search(regex, dep).group() for dep in pyproject}
            case dict():  # assume format `package = "<comparator>version"`
                deps = set(pyproject.keys())
            case _:
                raise TypeError(f"Unexpected type: {type(pyproject)}")

    return deps


def get_deps_pyproject_file(
    fname: str | Path = "pyproject.toml", /
) -> tuple[set[str], set[str]]:
    """Extract the dependencies from a pyproject.toml file.

    Returns:
        tuple of (project_dependencies, test_dependencies)

    There are 6 sections we check:
    - pyproject.dependencies
    - pyproject.optional-dependencies.test(s)
    - tool.poetry.dependencies
    - tool.poetry.group.test(s).dependencies

    If dependencies are specified in multiple sections, it is validated that they are
    the same.
    """
    with open(fname, "rb") as file:
        toml = tomllib.load(file)

    dependencies = {
        key: get_deps_pyproject_section(toml, key)
        for key in (
            "project.dependencies",
            "project.optional-dependencies.test",
            "project.optional-dependencies.tests",
            "tool.poetry.dependencies",
            "tool.poetry.group.test.dependencies",
            "tool.poetry.group.tests.dependencies",
        )
    }

    match (
        dependencies["project.dependencies"],
        dependencies["tool.poetry.dependencies"],
    ):
        case set() as a, set() as b:
            if (left := a - b) | (right := b - a):
                raise ValueError(
                    "Found different dependencies in [project] and [tool.poetry]."
                    f"\n [project]     is missing: {left}, "
                    f"\n [tool.poetry] is missing: {right}."
                )
            project_dependencies = a
        case set() as a, _:
            project_dependencies = a
        case _, set() as b:
            project_dependencies = b
        case _:
            project_dependencies = set()

    match (
        dependencies["project.optional-dependencies.test"],
        dependencies["project.optional-dependencies.tests"],
    ):
        case set() as a, set() as b:
            raise ValueError(
                "Found both [project.optional-dependencies.test] and [project.optional-dependencies.tests]."
            )
        case set() as a, _:
            project_test_dependencies = a
        case _, set() as b:
            project_test_dependencies = b
        case _:
            project_test_dependencies = NotImplemented

    match (
        dependencies["tool.poetry.group.test.dependencies"],
        dependencies["tool.poetry.group.tests.dependencies"],
    ):
        case set() as a, set() as b:
            raise ValueError(
                "Found both [tool.poetry.group.test.dependencies] and [tool.poetry.group.tests.dependencies]."
            )
        case set() as a, _:
            poetry_test_dependencies = a
        case _, set() as b:
            poetry_test_dependencies = b
        case _:
            poetry_test_dependencies = NotImplemented

    match (
        project_test_dependencies,
        poetry_test_dependencies,
    ):
        case set() as a, set() as b:
            if (left := a - b) | (right := b - a):
                raise ValueError(
                    "Found different test dependencies in [project] and [tool.poetry]."
                    f"\n [project]     is missing: {left}, "
                    f"\n [tool.poetry] is missing: {right}."
                )
            test_dependencies = a
        case set() as a, _:
            test_dependencies = a
        case _, set() as b:
            test_dependencies = b
        case _:
            test_dependencies = set()

    return project_dependencies, test_dependencies


def group_dependencies(dependencies: set[str], /) -> tuple[list[str], list[str]]:
    """Splits the dependencies into first-party and third-party."""
    stdlib_dependencies = set()
    third_party_dependencies = set()

    for dependency in dependencies:
        if dependency in sys.stdlib_module_names:
            stdlib_dependencies.add(dependency)
        else:
            third_party_dependencies.add(dependency)

    return sorted(stdlib_dependencies), sorted(third_party_dependencies)


def collect_dependencies(fname: str | Path, /, raise_notfound: bool = True) -> set[str]:
    """Collect the third-party dependencies from files in the given path."""
    path = Path(fname)
    dependencies = set()
    if not path.exists():  # assume module
        dependencies |= get_deps_module(str(fname))
    elif path.is_file():  # Single file
        dependencies |= get_deps_file(path)
    elif path.is_dir():  # Directory
        for file_path in path.rglob("*.py"):
            if file_path.is_file():
                dependencies |= get_deps_file(file_path)
    elif raise_notfound:
        raise FileNotFoundError(f"Invalid path: {path}")

    return dependencies


def main() -> None:
    """Print the third-party dependencies of a module."""
    # usage
    modules_default = ["src/"]
    tests_default = ["tests/"]

    parser = argparse.ArgumentParser(
        description="Print the third-party dependencies of a module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "modules",
        nargs="*",
        default=modules_default,
        type=str,
        help="The folder of the module to check.",
    )
    parser.add_argument(
        "--pyproject_file",
        default="pyproject.toml",
        type=str,
        help="The path to the pyproject.toml file.",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=tests_default,
        type=str,
        help="The path to the test directories.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Disables silencing of import messages.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information.",
    )
    args = parser.parse_args()

    modules_given = args.modules is not modules_default
    project_dependencies = set().union(
        *(
            collect_dependencies(fname, raise_notfound=modules_given)
            for fname in args.modules
        )
    )

    tests_given = args.tests is not tests_default
    test_dependencies = set().union(
        *(
            collect_dependencies(fname, raise_notfound=tests_given)
            for fname in args.tests
        )
    )

    _, project_deps_sorted = group_dependencies(project_dependencies)
    _, test_deps_sorted = group_dependencies(test_dependencies - project_dependencies)

    for dependency in project_deps_sorted:
        print(dependency)

    print("additional test dependencies:")

    for dependency in test_deps_sorted:
        print(dependency)


if __name__ == "__main__":
    main()
    # print(get_deps_pyproject_file())
    # print(collect_dependencies("src/"))
    # print(collect_dependencies("tests/"))


# def is_submodule(submodule_name: str, module_name: str, /) -> bool:
#     """True if submodule_name is a submodule of module_name."""
#     return submodule_name.startswith(f"{module_name}.")
