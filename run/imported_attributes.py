#!/usr/bin/env python
"""Check whether attributes in annotations shadow directly imported symbols.

Example:
    >>> import pandas as pd
    >>> from pandas import DataFrame
    >>>
    >>> def foo(df: pd.DataFrame) -> pd.DataFrame:
    >>>     return df

    Would raise an error because `pd.DataFrame` shadows directly imported `DataFrame`.
"""

import argparse
import ast
import sys
from collections.abc import Iterator
from pathlib import Path


def get_attributes(tree: ast.AST, /) -> Iterator[ast.Attribute]:
    """Get all attribute nodes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            yield node


def get_type_hints(tree: ast.AST, /) -> Iterator[ast.AST]:
    """Get all nodes that are type hints."""
    for node in ast.walk(tree):
        ann = getattr(node, "annotation", None)
        if ann is not None:
            yield ann


def get_attrs_shadow_imported(tree: ast.AST, /) -> Iterator[ast.Attribute]:
    """Get attribute nodes that shadow directly imported symbols."""
    imported_symbols = get_imported_symbols(tree)

    # for node in get_type_hints(tree):
    for node in get_attributes(tree):
        if node.attr in imported_symbols:
            yield node


def get_full_attribute_string(node: ast.Attribute, /) -> str:
    """Get the parent of an attribute node."""
    if isinstance(node.value, ast.Attribute):
        return get_full_attribute_string(node.value) + "." + node.attr

    if not isinstance(node.value, ast.Name):
        raise ValueError(f"Expected ast.Name, got {type(node.value)}")

    return node.value.id + "." + node.attr


def get_full_attribute_parent(node: ast.Attribute, /) -> tuple[ast.Name, str]:
    """Get the parent of an attribute node."""
    if isinstance(node, ast.Attribute):
        parent, string = get_full_attribute_parent(node.value)
        return parent, f"{string}.{node.attr}"

    if not isinstance(node, ast.Name):
        raise ValueError(f"Expected ast.Name, got {type(node.value)}")

    return node, node.id


def get_imported_symbols(tree: ast.AST, /) -> dict[str, str]:
    """Get all imported symbols."""
    imported_symbols = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_symbols[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            if module_name is not None:
                for alias in node.names:
                    full_name = f"{module_name}.{alias.name}"
                    imported_symbols[alias.asname or alias.name] = full_name

    return imported_symbols


def get_imported_attributes(
    tree: ast.AST, /, debug: bool = False
) -> Iterator[tuple[ast.Attribute, ast.Name, str]]:
    """Finds attributes that can be replaced by directly imported symbols."""
    imported_symbols = get_imported_symbols(tree)

    for node in get_attributes(tree):
        if node.attr in imported_symbols:
            # parent = get_full_attribute_string(node)
            parent, string = get_full_attribute_parent(node)

            head, tail = string.split(".", maxsplit=1)
            assert head == parent.id

            # e.g. DataFrame -> pandas.DataFrame
            matched_symbol = imported_symbols[node.attr]
            is_match = matched_symbol == string

            # need to check if parent is imported as well to catch pd.DataFrame
            if parent.id in imported_symbols:
                parent_alias = imported_symbols[parent.id]  # e.g. pd -> pandas
                is_match |= matched_symbol == f"{parent_alias}.{tail}"

            if is_match:
                yield node, parent, string


def detect_in_file(file_path: Path, /, debug: bool = False) -> bool:
    """Finds shadowed attributes in a file."""
    # Your code here
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    # find all violations
    node: ast.Attribute = NotImplemented
    for node, _, string in get_imported_attributes(tree):
        print(
            f"{file_path!s}:{node.lineno!s}"
            f" use directly imported {node.attr!r} instead of {string!r}"
        )
    passed = node is NotImplemented

    if not passed and debug:
        imported_symbols = get_imported_symbols(tree)
        pad = " " * 4
        max_key_len = max(len(key) for key in imported_symbols.keys())
        print(pad, "Imported symbols:")
        for key, value in imported_symbols.items():
            print(2 * pad, f"{key:{max_key_len}} -> {value}")

    return passed


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Apply a script to files matched by fnmatch patterns in the current working directory."
    )
    parser.add_argument(
        "files", nargs="+", help="One or multiple files or file patterns."
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # find all files
    root = Path.cwd().absolute()
    files: list[Path] = []
    for file_or_pattern in args.files:
        # case: file
        path = Path(file_or_pattern).absolute()
        if path.exists():
            files.append(path)
            continue
        else:
            matches = list(root.glob(file_or_pattern))
            if not matches:
                raise FileNotFoundError(
                    f"Pattern {file_or_pattern!r} did not match any files."
                )
            files.extend(matches)

    # apply script to all files
    passed = True
    for file in files:
        passed &= detect_in_file(file, debug=args.debug)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
