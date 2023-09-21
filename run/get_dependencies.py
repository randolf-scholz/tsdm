#!/usr/bin/env python
"""Prints the direct dependencies of a module line by line.."""

# TODO: add support for extras.

__all__ = [
    "collect_dependencies",
    "get_deps_file",
    "get_deps_import",
    "get_deps_importfrom",
    "get_deps_module",
    "get_deps_pyproject",
    "get_deps_pyproject_section",
    "get_deps_pyproject_test",
    "get_deps_tree",
    "group_dependencies",
    "main",
    "normalize_dep_name",
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
from typing import Any, NamedTuple

if sys.version_info >= (3, 10):
    import importlib.metadata as metadata
else:
    try:
        metadata = importlib.import_module("importlib_metadata")
    except ImportError as E:
        raise ImportError(
            "This pre-commit hook runs in the local interpreter and requires"
            " the `importlib_metadata` package for python versions < 3.10."
        ) from E

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        tomllib = importlib.import_module("tomlkit")
    except ImportError as E:
        raise ImportError(
            "This pre-commit hook runs in the local interpreter and requires"
            " the `tomlkit` package for python versions < 3.11."
        ) from E

PACKAGES: dict[str, list[str]] = (
    metadata.packages_distributions()
)  # type:ignore[assignment]
"""A dictionary that maps module names to their pip-package names."""

# NOTE: illogical type hint in stdlib, maybe open issue.
# https://github.com/python/cpython/blob/608927b01447b110de5094271fbc4d49c60130b0/Lib/importlib/metadata/__init__.py#L933-L947C29
# https://github.com/python/typeshed/blob/d82a8325faf35aa0c9d03d9e9d4a39b7fcb78f8e/stdlib/importlib/metadata/__init__.pyi#L32


def normalize_dep_name(dep: str, /) -> str:
    """Normalize a dependency name."""
    return dep.lower().replace("-", "_")


def get_deps_import(node: ast.Import, /) -> set[str]:
    """Extract dependencies from an `import ...` statement."""
    return {alias.name.split(".")[0] for alias in node.names}
    # dependencies = set()
    # for alias in node.names:
    #     module = alias.name.split(".")[0]
    #     if not module.startswith("_"):
    #         dependencies.add(module)
    # return dependencies


def get_deps_importfrom(node: ast.ImportFrom, /) -> set[str]:
    """Extract dependencies from an `from y import ...` statement."""
    assert node.module is not None
    module_name = node.module.split(".")[0]
    if module_name.startswith("_"):  # ignore _private modules
        return set()
    return {module_name}


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
    assert module.__file__ is not None
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


def get_deps_pyproject_section(config: dict[str, Any], /, *, section: str) -> set[str]:
    """Get the dependencies from a section of pyproject.toml.

    Looking up the section must either result in a list of strings or a dict.
    """
    try:  # recursively get the section
        for key in section.split("."):
            config = config[key]
    except KeyError:
        return NotImplemented

    match config:
        case list() as lst:  # type: ignore[unreachable]
            # assume format `"package<comparator>version"`
            regex = re.compile(r"[a-zA-Z0-9_-]*")  # type: ignore[unreachable]
            return {re.search(regex, dep).group() for dep in lst}
        case dict() as dct:  # poetry
            # assume format `package = "<comparator>version"`
            return set(dct.keys()) - {"python"}
        case _:
            raise TypeError(f"Unexpected type: {type(config)}")


def get_deps_pyproject(fname: str | Path = "pyproject.toml", /) -> set[str]:
    """Extract the dependencies from a pyproject.toml file.

    There are 6 sections we check:
    - pyproject.dependencies
    - pyproject.optional-dependencies.test(s)
    - tool.poetry.dependencies
    - tool.poetry.group.test(s).dependencies

    If dependencies are specified in multiple sections, it is validated that they are
    the same.
    """
    with open(fname, "rb") as file:
        pyproject = tomllib.load(file)

    dependencies = {
        key: get_deps_pyproject_section(pyproject, section=key)
        for key in (
            "project.dependencies",
            "tool.poetry.dependencies",
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

    return project_dependencies


def get_deps_pyproject_test(fname: str | Path = "pyproject.toml", /) -> set[str]:
    """Extract the test dependencies from a pyproject.toml file."""
    with open(fname, "rb") as file:
        pyproject = tomllib.load(file)

    dependencies = {
        key: get_deps_pyproject_section(pyproject, section=key)
        for key in (
            "project.optional-dependencies.test",
            "project.optional-dependencies.tests",
            "tool.poetry.group.test.dependencies",
            "tool.poetry.group.tests.dependencies",
        )
    }

    match (
        dependencies["project.optional-dependencies.test"],
        dependencies["project.optional-dependencies.tests"],
    ):
        case set(), set():
            raise ValueError(
                "Found both [project.optional-dependencies.test]"
                " and [project.optional-dependencies.tests]."
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
        case set(), set():
            raise ValueError(
                "Found both [tool.poetry.group.test.dependencies]"
                " and [tool.poetry.group.tests.dependencies]."
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

    return test_dependencies


class GroupedDependencies(NamedTuple):
    """A named tuple containing the dependencies grouped by type."""

    imported_dependencies: set[str]
    stdlib_dependencies: set[str]


def group_dependencies(dependencies: set[str], /) -> GroupedDependencies:
    """Splits the dependencies into first-party and third-party."""
    imported_dependencies = set()
    stdlib_dependencies = set()

    for dependency in dependencies:
        if dependency in sys.stdlib_module_names:
            stdlib_dependencies.add(dependency)
        else:
            imported_dependencies.add(dependency)

    return GroupedDependencies(
        imported_dependencies=imported_dependencies,
        stdlib_dependencies=stdlib_dependencies,
    )


def collect_dependencies(fname: str | Path, /, raise_notfound: bool = True) -> set[str]:
    """Collect the third-party dependencies from files in the given path."""
    path = Path(fname)
    dependencies = set()

    if path.is_file():  # Single file
        dependencies |= get_deps_file(path)
    elif path.is_dir():  # Directory
        for file_path in path.rglob("*.py"):
            if file_path.is_file():
                dependencies |= get_deps_file(file_path)
    elif not path.exists():  # assume module
        try:
            dependencies |= get_deps_module(str(fname))
        except ModuleNotFoundError as exc:
            if raise_notfound:
                raise exc
    elif raise_notfound:
        raise FileNotFoundError(f"Invalid path: {path}")

    return dependencies


def validate_dependencies(
    *,
    pyproject_dependencies: set[str],
    imported_dependencies: set[str],
    raise_unused_dependencies: bool = True,
) -> None:
    """Validate the dependencies."""
    # extract 3rd party dependencies.
    used_deps = group_dependencies(imported_dependencies).imported_dependencies

    # map the dependencies to their pip-package names
    imported_deps: set[str] = set()
    unknown_deps: set[str] = set()
    for dep in used_deps:
        if dep not in PACKAGES:
            unknown_deps.add(dep)
            continue

        # get the pypi-package name
        values: list[str] = PACKAGES[dep]
        if len(values) > 1:
            raise ValueError(f"Found multiple pip-packages for {dep!r}: {values}.")
        imported_deps.add(values[0])

    # normalize the dependencies
    pyproject_deps = {normalize_dep_name(dep) for dep in pyproject_dependencies}
    imported_deps = {normalize_dep_name(dep) for dep in imported_deps}

    # check if all imported dependencies are listed in pyproject.toml
    missing_deps = imported_deps - pyproject_deps
    unused_deps = pyproject_deps - imported_deps

    if missing_deps or unknown_deps or (unused_deps and raise_unused_dependencies):
        raise ValueError(
            "Found discrepancy between imported dependencies and pyproject.toml!"
            f"\nImported dependencies not listed in pyproject.toml: {missing_deps}."
            f"\nUnused dependencies listed in pyproject.toml: {unused_deps}."
            f"\nUnknown dependencies: {unknown_deps}."
            "\n"
            "\nOptional dependencies are currently not supported (PR welcome)."
            "\nWorkaround: use `importlib.import_module('optional_dependency')`."
        )


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
        "pyproject_file",
        nargs="?",
        default="pyproject.toml",
        type=str,
        help="The path to the pyproject.toml file.",
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        default=modules_default,
        type=str,
        help="The folder of the module to check.",
    )
    parser.add_argument(
        "--error_unused_project_deps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise error if pyproject.toml lists unused project dependencies",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=tests_default,
        type=str,
        help="The path to the test directories.",
    )
    parser.add_argument(
        "--error_unused_test_deps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise error if pyproject.toml lists unused test dependencies",
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

    # compute the dependencies from the source files
    modules_given = args.modules is not modules_default
    imported_dependencies = set().union(
        *(
            collect_dependencies(fname, raise_notfound=modules_given)
            for fname in args.modules
        )
    )
    # get dependencies from pyproject.toml
    pyproject_dependencies = get_deps_pyproject(args.pyproject_file)
    # validate the dependencies
    validate_dependencies(
        pyproject_dependencies=pyproject_dependencies,
        imported_dependencies=imported_dependencies,
    )

    # compute the test dependencies from the test files
    tests_given = args.tests is not tests_default
    imported_test_dependencies = set().union(
        *(
            collect_dependencies(fname, raise_notfound=tests_given)
            for fname in args.tests
        )
    )
    # get dependencies from pyproject.toml
    pyproject_test_dependencies = get_deps_pyproject(args.pyproject_file)
    # validate the dependencies
    validate_dependencies(
        pyproject_dependencies=pyproject_test_dependencies,
        imported_dependencies=imported_test_dependencies,
    )


if __name__ == "__main__":
    main()
