# AGENTS.md

This file describes project conventions for automated agents contributing to `tsdm`.

## Project snapshot

- **Language**: Python
- **Packaging**: `pyproject\.toml` (PEP 621)
- **Target runtime**: Python `>=3.14`, `<3.15` on Linux

## Code Style

- Prefer `match` statements over long `if-elif-else` when applicable.
- avoid deeply nested if-else blocks
- Use f-strings for string formatting.
- Prefer `list`/`dict`/`set` comprehensions over `for` loops and `map`/`filter` constructs.
- Use context managers (`with` statements) for resource management (e.g., file handling, database connections).

## Type hinting policy

### Annotation style

- Type annotations are checked with `pyright` and `mypy`.
- Use Python 3.12+ annotation syntax:
  - Use `type Alias = ...` instead of `Alias: TypeAlias = ...`.
  - Use `collections.abc` members instead of `typing` equivalents (`Iterable`, `Sequence`, `Mapping`, etc.)
  - Use builtin generics: `list[int]`, `dict[str, int]`, etc. instead of `List[int]`, `Dict[...]`, etc.
  - Use `T | U` instead of `Union[T, U]` (exceptions apply to `Optional[T]`)
- Use `Optional[T]` for optional keyword parameters that can be `None`, and for return types that may be `None`. In
  other cases, use `T | None`.
- Prefer loose types for function parameters:
  - If a function argument expects a list-like input, prefer `Sequence[T]` over `list[T]`.
  - If a function argument expects a dict-like input, prefer `Mapping[K, V]` over `dict[K, V]`.
  - If a function argument expects a set-like input, prefer `AbstractSet[T]` over `set[T]`.
  - Use `MutableSequence[T]`/`MutableMapping[K, V]`/`MutableSet[T]` only when mutation is required.
- Prefer precise types for return values:
  - If a function returns a list, use `list[T]` instead of `Sequence[T]`.
  - Avoid returning `Union` type, except for `Optional[T]` and `@overload` implementations.
  - Abstract base classes should return general types (e.g., `Sequence[T]`), while concrete implementations should
    return specific types (e.g., `list[T]`).

## Function signatures

- If a parameter name is not semantically meaningful prefer `POSITION_ONLY` parameters (use `/`).
- Avoid `POSITIONAL_OR_KEYWORD` parameters for the most part. Limit their use to at most `2` parameters per function.
- If a function expects `*args`, then no `POSITIONAL_OR_KEYWORD` arguments are allowed.
- Additional parameters should generally be `KEYWORD_ONLY` (use `*`).

## Formatting and linting

- Keep code compatible with the configured tooling in `pyproject.toml` (ruff, mypy, pyright, pylint).
- Match existing project formatting (line length, imports, quotes) and do not introduce conflicting style rules.

## Documentation expectations

- Docstrings should align with the configured convention (Google style).
- When behavior is subtle, document invariants and edge cases alongside the type signature.

## Running tests, formatting, and linting

- use `uv run <tool>` for all project-local Python commands, including `python`, `pytest`, `ruff`, `pyrefly`, and
  `pyright`.

## Shell commands and approvals

- Run read-only inspection commands separately rather than combining them with
  `;`, `&&`, or other shell composition when there is no dependency between them.
- In particular, run `sed -n`, `rg -n`, `rg --files`, and read-only `git`
  commands such as `git diff`, `git status`, `git log`, and `git show`
  as individual commands.
- These commands are already permitted by the execution policy. Do not request elevated permissions or additional
  approval for them.
- Only combine commands when their execution genuinely depends on the previous command succeeding.
- If `apply_patch` is failing, try `git apply` or `patch` as fallbacks.
