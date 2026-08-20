# Contributing to `tsdm`

Thank you for contributing to `tsdm`. The project targets Python 3.14 on
Linux and uses [`uv`](https://docs.astral.sh/uv/) to manage the virtual
environment, Python version, dependencies, and development tools.

## Set up the repository

Fork <https://github.com/randolf-scholz/tsdm> on GitHub, then clone your fork:

```bash
git clone https://github.com/$USER/tsdm.git
cd tsdm
git submodule update --init --recursive
```

Create the virtual environment, install the project and its development
dependencies, and install the Git hooks:

```bash
uv python install 3.14
uv sync
uv run prek install
```

`uv sync` creates `.venv` when necessary. You do not need to activate it:
run project commands with `uv run` so that they always use the managed
environment.

Verify the installation:

```bash
uv run python -c "import tsdm"
```

## Make a change

Create a descriptively named branch:

```bash
git switch -c feature-xyz
```

Source code lives in `src/`, and tests live in `tests/`. Add or update tests
for behavior affected by your change.

### Run the checks

Run the test suite with `pytest`:

```bash
uv run pytest
```

During development, you can pass a file, directory, or pytest selector to run
a focused subset:

```bash
uv run pytest tests/path/to/test_module.py
```

Lint and check formatting with `ruff`:

```bash
uv run ruff check .
uv run ruff format --check .
```

To apply safe lint fixes and formatting:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Run both supported type checkers:

```bash
uv run pyrefly check
uv run pyright
```

Run all configured `prek` hooks across the repository before committing:

```bash
uv run prek run --all-files
```

The installed Git hook also runs the applicable checks automatically when you
commit.

## Submit the change

Keep commits focused and use descriptive commit messages. Push your branch to
your fork:

```bash
git push -u origin feature-xyz
```

Confirm that the continuous-integration checks pass, then open a pull request
against the `main` branch of <https://github.com/randolf-scholz/tsdm>.
