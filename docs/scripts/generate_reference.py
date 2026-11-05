r"""Generate Zensical API pages for public modules, classes, and functions."""

import ast
import posixpath
import shutil
from collections.abc import Collection, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import mkdocs_gen_files

ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "src"
REFERENCE_DIR = Path("reference")
REFERENCE_INDEX = REFERENCE_DIR / "index.md"


@dataclass(frozen=True)
class Module:
    r"""Source and public API metadata for one Python module."""

    identifier: str
    doc_path: Path
    exports: Sequence[str]
    is_package: bool
    kinds: Mapping[str, str]
    nodes: Sequence[ast.stmt]
    source_path: Path


def write_page(
    identifier: str,
    doc_path: Path,
    *,
    source_path: Path,
    nav_parts: tuple[str, ...],
    symbol_type: str,
    members: Sequence[str] = (),
    nav: mkdocs_gen_files.Nav,
    documented_identifiers: set[str],
) -> None:
    r"""Create an API page and navigation entry; empty `members` omit the override."""
    # Overload declarations share an identifier and must produce only one page.
    if identifier in documented_identifiers:
        return

    documented_identifiers.add(identifier)
    nav[nav_parts] = doc_path.relative_to(REFERENCE_DIR).as_posix()
    with mkdocs_gen_files.open(doc_path, "w") as file:
        title = identifier.rsplit(".", 1)[-1]
        file.write(
            f"---\ntitle: {title}\nsymbol_type: {symbol_type}\n---\n\n"
            f"::: {identifier}\n"
        )
        if members:
            file.write(
                "    options:\n      members:\n"
                + "".join(f"        - {name!r}\n" for name in members)
            )
    mkdocs_gen_files.set_edit_path(doc_path, source_path.relative_to(ROOT_DIR))


def get_exports(nodes: Sequence[ast.stmt]) -> list[str]:
    r"""Return the names declared by a literal module `__all__`."""
    for node in nodes:
        match node:
            case ast.Assign(targets=targets, value=value) if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in targets
            ):
                try:
                    names = ast.literal_eval(value)
                except ValueError:
                    return []
                return [name for name in names if isinstance(name, str)]
    return []


def get_kinds(nodes: Sequence[ast.stmt], exports: Collection[str]) -> dict[str, str]:
    r"""Classify directly declared public classes, functions, and constants."""
    kinds: dict[str, str] = {}
    for node in nodes:
        match node:
            case ast.ClassDef(name=name) if not name.startswith("_"):
                kinds[name] = "classes"

            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) if (
                not name.startswith("_")
            ):
                kinds[name] = "functions"

            case ast.Assign(targets=targets):
                names = [
                    target.id for target in targets if isinstance(target, ast.Name)
                ]
                for name in names:
                    if name in exports or (not name.startswith("_") and name.isupper()):
                        kinds[name] = "constants"

            case (
                ast.AnnAssign(target=ast.Name(id=name))
                | ast.TypeAlias(name=ast.Name(id=name))
            ):
                if name in exports or (not name.startswith("_") and name.isupper()):
                    kinds[name] = "constants"
    return kinds


def get_class_members(nodes: Sequence[ast.stmt]) -> list[str]:
    r"""Return the public methods and attributes declared directly by a class."""
    members: list[str] = []
    for node in nodes:
        match node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                if name == "__init__" or not name.startswith("_"):
                    members.append(name)
            case ast.Assign(targets=targets):
                members.extend(
                    target.id
                    for target in targets
                    if isinstance(target, ast.Name) and not target.id.startswith("_")
                )
            case ast.AnnAssign(target=ast.Name(id=name)) | ast.ClassDef(name=name):
                if not name.startswith("_"):
                    members.append(name)
    return members


def get_modules(source_dir: Path, /) -> dict[str, Module]:
    r"""Discover public modules, excluding private paths and `__main__` scripts."""
    modules: dict[str, Module] = {}
    for source_path in sorted(source_dir.rglob("*.py")):
        module_parts = source_path.relative_to(source_dir).with_suffix("").parts

        # Package __init__.py files represent their containing package.
        is_package = module_parts[-1] == "__init__"
        if is_package:
            module_parts = module_parts[:-1]

        if not module_parts or any(part.startswith("_") for part in module_parts):
            continue

        # Inspect source without importing modules or running their initialization.
        identifier = ".".join(module_parts)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        exports = get_exports(tree.body)
        modules[identifier] = Module(
            identifier=identifier,
            doc_path=REFERENCE_DIR.joinpath(*module_parts, "index.md"),
            exports=exports,
            is_package=is_package,
            kinds=get_kinds(tree.body, exports),
            nodes=tree.body,
            source_path=source_path,
        )
    return modules


def get_page_paths(modules: Mapping[str, Module], /) -> dict[str, Path]:
    r"""Map declared classes and functions to their generated documentation paths."""
    page_paths: dict[str, Path] = {}
    for module in modules.values():
        for name, kind in module.kinds.items():
            match kind:
                case "classes":
                    path = module.doc_path.parent / name / "index.md"
                case "functions":
                    path = module.doc_path.parent / f"{name}.md"
                case _:
                    continue
            page_paths[f"{module.identifier}.{name}"] = path
    return page_paths


def get_imports(module: Module) -> dict[str, str]:
    r"""Return fully-qualified targets for names imported by a module."""
    package_parts = module.identifier.split(".")
    if not module.is_package:
        package_parts = package_parts[:-1]

    imports: dict[str, str] = {}
    for node in module.nodes:
        if not isinstance(node, ast.ImportFrom):
            continue

        target_parts = package_parts[: len(package_parts) - node.level + 1]
        if node.module:
            target_parts.extend(node.module.split("."))

        target_module = ".".join(target_parts)
        for alias in node.names:
            if alias.name != "*":
                imports[alias.asname or alias.name] = f"{target_module}.{alias.name}"
    return imports


def get_target(
    module: Module,
    name: str,
    *,
    modules: Mapping[str, Module],
    page_paths: Mapping[str, Path],
    seen: AbstractSet[str] = frozenset(),
) -> str | None:
    r"""Resolve a canonical identifier without mutating `seen`; return `None` on cycles."""
    identifier = f"{module.identifier}.{name}"
    if identifier in page_paths:
        return identifier

    if identifier in seen:
        return None

    # Follow re-exports until a declaration or an import cycle is reached.
    target = get_imports(module).get(name)
    if target is None:
        return None
    if target in page_paths:
        return target

    target_module, _, target_name = target.rpartition(".")
    if target_module in modules:
        return get_target(
            modules[target_module],
            target_name,
            modules=modules,
            page_paths=page_paths,
            seen=seen | {identifier},
        )
    return None


def get_relative_path(source: Path, target: Path) -> str:
    r"""Return the relative path from one generated document to another."""
    return posixpath.relpath(target.as_posix(), start=source.parent.as_posix())


def write_contents(
    module: Module,
    file: TextIO,
    *,
    modules: Mapping[str, Module],
    page_paths: Mapping[str, Path],
) -> None:
    r"""Write a module's public API overview and constants documentation."""
    categories = {"classes": [], "functions": [], "submodules": [], "constants": []}
    # Prefer explicit exports; otherwise use public declarations in source order.
    names = module.exports or list(module.kinds)
    for name in names:
        target = get_target(module, name, modules=modules, page_paths=page_paths)
        if name in module.kinds:
            kind = module.kinds[name]
        elif target is not None:
            target_module, _, target_name = target.rpartition(".")
            kind = modules[target_module].kinds[target_name]
        else:
            kind = "constants"

        if target is None:
            entry = f"`{name}`"
        else:
            path = get_relative_path(module.doc_path, page_paths[target])
            entry = f"[{name}]({path})"

        categories[kind].append(entry)

    # Include immediate child modules even when they are absent from __all__.
    for identifier, submodule in modules.items():
        parent, _, name = identifier.rpartition(".")
        if parent == module.identifier:
            path = get_relative_path(module.doc_path, submodule.doc_path)
            categories["submodules"].append(f"[{name}]({path})")

    if not any(categories.values()):
        return

    file.write("\n## Contents\n")
    for heading, entries in categories.items():
        if entries:
            file.write(
                f"\n### {heading.title()}\n\n"
                + "".join(f"- {entry}\n" for entry in entries)
            )

    # Constants stay on the module page rather than receiving individual pages.
    constants = [name for name, kind in module.kinds.items() if kind == "constants"]
    if constants:
        file.write(
            "\n## Constants\n\n"
            f"::: {module.identifier}\n"
            "    options:\n"
            "      heading_level: 3\n"
            "      show_docstring_description: false\n"
            "      show_root_heading: false\n"
            "      show_root_toc_entry: false\n"
            "      members:\n" + "".join(f"        - {name!r}\n" for name in constants)
        )


def document_module_definitions(
    module: Module,
    *,
    page_paths: Mapping[str, Path],
    nav: mkdocs_gen_files.Nav,
    documented_identifiers: set[str],
) -> None:
    r"""Create pages for public module-level classes and functions."""
    positions = {name: index for index, name in enumerate(module.exports)}
    defs = [
        node
        for node in module.nodes
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]

    # Stable sorting keeps definitions absent from __all__ in source order.
    for node in sorted(defs, key=lambda n: positions.get(n.name, len(positions))):
        match node:
            case ast.ClassDef(body=body):
                symbol_type = "class"
                members = get_class_members(body)
            case _:
                symbol_type = "function"
                members = ()
        identifier = f"{module.identifier}.{node.name}"
        write_page(
            identifier,
            page_paths[identifier],
            source_path=module.source_path,
            nav_parts=(*module.identifier.split("."), node.name),
            symbol_type=symbol_type,
            members=members,
            nav=nav,
            documented_identifiers=documented_identifiers,
        )


def main() -> None:
    r"""Stage authored Markdown and assets, then generate the public API pages."""
    # Zensical reads only this staging directory, keeping Sphinx tooling out.
    docs_dir = ROOT_DIR / "docs"
    staging_dir = mkdocs_gen_files.DOCS_DIR
    staging_dir.mkdir(parents=True, exist_ok=True)
    for path in docs_dir.glob("*.md"):
        shutil.copy2(path, staging_dir / path.name)
    shutil.copytree(
        docs_dir / "_static",
        staging_dir / "_static",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.rst"),
    )

    # Discover all targets before writing pages so forward references can resolve.
    modules = get_modules(SOURCE_DIR)
    page_paths = get_page_paths(modules)
    nav = mkdocs_gen_files.Nav()
    documented_identifiers: set[str] = set()

    # Generate each module's overview, then its class and function pages.
    for module in modules.values():
        write_page(
            module.identifier,
            module.doc_path,
            source_path=module.source_path,
            nav_parts=tuple(module.identifier.split(".")),
            symbol_type="module",
            nav=nav,
            documented_identifiers=documented_identifiers,
        )
        with mkdocs_gen_files.open(module.doc_path, "a") as file:
            write_contents(module, file, modules=modules, page_paths=page_paths)
        document_module_definitions(
            module,
            page_paths=page_paths,
            nav=nav,
            documented_identifiers=documented_identifiers,
        )

    # The reference landing page lists only top-level packages.
    with mkdocs_gen_files.open(REFERENCE_INDEX, "w") as file:
        file.write("# API reference\n\n## Packages\n\n")
        for module in modules.values():
            if "." not in module.identifier:
                path = get_relative_path(REFERENCE_INDEX, module.doc_path)
                file.write(f"- [{module.identifier}]({path})\n")

    # Literate navigation mirrors the generated page hierarchy.
    with mkdocs_gen_files.open(REFERENCE_DIR / "SUMMARY.md", "w") as file:
        file.write("- [API reference](index.md)\n" + "".join(nav.build_literate_nav()))


if __name__ == "__main__":
    main()
