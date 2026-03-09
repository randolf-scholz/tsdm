r"""Utility to generate string variants."""

from collections.abc import Callable


def variants(s: str | list[str], /) -> list[str]:
    r"""Return all variants of a string."""
    if isinstance(s, str):
        cases: list[Callable[[str], str]] = [
            lambda x: x.lower(),
            lambda x: x.capitalize(),
            lambda x: x.upper(),
        ]
        decorations: list[Callable[[str], str]] = [
            lambda x: x,
            lambda x: f"#{x}",
            lambda x: f"<{x}>",
            lambda x: f"+{x}",
            lambda x: f"-{x}",
        ]
        return [deco(case(s)) for deco in decorations for case in cases]
    # return concatenation of all variants
    return [j for i in (variants(s_) for s_ in s) for j in i]
