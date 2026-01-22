# AGENTS.md

This file describes project conventions for automated agents contributing to `tsdm`.

## Project snapshot

- **Language**: Python
- **Packaging**: `pyproject\.toml` (PEP 621)
- **Target runtime**: Python `>=3.14`, `<3.15`

## Python compatibility

- Use the most recent Python version: **3.14**.
- Prefer Python 3.14 standard library features over third-party backports.

## Type hinting policy

### Annotation style

- Type annotations are checked with `mypy` and `pyright`, with preference to `pyright` behavior in case of conflicts.
- Use Python 3.12+ annotation syntax (PEP 695) where applicable:
  - Prefer `type Alias = ...` over `Alias: TypeAlias = ...`.
  - Prefer generic functions/classes using PEP 695 syntax when it improves clarity.
- Prefer builtin generics: `list[int]`, `dict[str, int]`, etc. instead of `List[int]`, `Dict[...]`, etc.
- Prefer `collections.abc` protocols over `typing` names when applicable.
- Prefer **general** input types for function parameters:
  Use `Iterable[T]`/`Sequence[T]`/`Mapping[K, V]` instead of `list[T]`/`dict[K, V]`
  unless mutability or specific operations are required.
- Prefer precise return types:
  - Avoid returning `Union` types by default.
    - Exception 1: returning `Optional[T]` is acceptable when `None` is a valid return value.
    - Exception 2: `@overload` implementations may return unions internally,
      but exported (public) function signatures should be as specific as possible.
  - Abstract base classes should return general types (e.g., `Sequence[T]`),
    while concrete implementations should return specific types (e.g., `list[T]`).

### Optional and `None`

- For function parameters: Use `Optional[T]` for optional keyword parameters when they can be `None`, and `T | None` otherwise.
- For return types: Prefer `Optional[T]` over `T | None` when the return value may be `None`.
- Never use implicit optional types (like `x: int = None`).

### Generics and invariance

- Use `Sequence[T]` for read-only containers and `MutableSequence[T]` only when mutation is required.
- Use `Mapping[K, V]` for read-only dict-like inputs and `MutableMapping[K, V]` only when mutation is required.
- Be mindful of variance: choose the widest safe abstraction for parameters.

## Function signatures

- Prefer at most `2` `POSITIONAL_OR_KEYWORD` parameters in public functions/methods.
  - Additional parameters should generally be `KEYWORD_ONLY\` (use `*`).
- If a parameter name is not semantically meaningful (i.e. the name is arbitrary and callers should not rely on it), make it `POSITION_ONLY\` (use `/`).
- Prefer stable, descriptive keyword names for public APIs; use positional-only primarily for protocol/duck-typed parameters or to preserve API flexibility.

## Formatting and linting

- Keep code compatible with the configured tooling in `pyproject.toml` (ruff, mypy, pyright, pylint).
- Match existing project formatting (line length, imports, quotes) and do not introduce conflicting style rules.

## Documentation expectations

- Docstrings should align with the configured convention (Google style).
- When behavior is subtle, document invariants and edge cases alongside the type signature.
