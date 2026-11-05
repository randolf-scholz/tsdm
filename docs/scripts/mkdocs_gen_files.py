r"""Small filesystem-backed subset of ``mkdocs_gen_files`` for Zensical builds."""

from pathlib import Path
from typing import TextIO

DOCS_DIR = Path(__file__).resolve().parents[1] / "build" / "markdown"
type Tree = dict[str, str | Tree]


class Nav:
    r"""Build a literate navigation file from assigned page paths."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, ...], str] = {}

    def __setitem__(self, parts: tuple[str, ...], path: str) -> None:
        self.entries[parts] = path

    def build_literate_nav(self) -> list[str]:
        tree: Tree = {}
        for parts, path in self.entries.items():
            current = tree
            for part in parts:
                child = current.setdefault(part, {})
                assert isinstance(child, dict)
                current = child
            current["__path__"] = path

        return self.render(tree)

    def render(self, nodes: Tree, level: int = 0) -> list[str]:
        lines: list[str] = []
        for name, node in nodes.items():
            if name == "__path__":
                continue
            assert isinstance(node, dict)
            path = node.get("__path__")
            label = f"[{name}]({path})" if isinstance(path, str) else name
            lines.append(f"{'    ' * level}- {label}\n")
            lines.extend(self.render(node, level + 1))
        return lines


def open(path: str | Path, mode: str = "r", **kwargs: object) -> TextIO:  # ruff: ignore[A001]
    r"""Open a generated document relative to the Zensical documentation root."""
    target = DOCS_DIR / path
    if any(flag in mode for flag in "wax+"):
        target.parent.mkdir(parents=True, exist_ok=True)
    return target.open(mode, **kwargs)  # type: ignore[return-value]


def set_edit_path(path: str | Path, edit_path: str | Path) -> None:
    r"""Accept MkDocs generator metadata unsupported by Zensical."""
