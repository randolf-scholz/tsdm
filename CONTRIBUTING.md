# CONTRIBUTING

These are the 10 steps to contributing to the project.

## 1. Fork the GitLab project from <https://github.com/randolf-scholz/tsdm>

Use your personal namespace, e.g. <https://github.com/$USER/tsdm>.

## 2. Clone the forked project locally to your machine

```bash
git clone https://github.com/randolf-scholz/tsdm
cd tsdm
```

### 3. Set up the virtual environment

Via `poetry` (recommended).

```bash
pip install --upgrade poetry
poetry shell
poetry install
```

Via `conda` (You may have to rename `tables` ⟶ `pytables` and `torch` ⟶ `pytorch`).

```bash
conda create --name tsdm --file requirements.txt
conda activate tsdm
conda install --file requirements-dev.txt
```

Via `pip`.

```bash
sudo apt install python3.10
python3.10 -m virtualenv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Verify that the installation was successful

```bash
python -c "import tsdm"
```

### 5. Setup remote repositories and pre-commit hooks

```bash
./run/setup_remote.sh
./run/setup_precommit.sh
```

### 6. Create a new working branch. Choose a descriptive name for what you are trying to achieve

```bash
git checkout -b feature-xyz
```

### 7. Write your code, bonus points for also adding unit tests

- Write your code in the `src` directory.
- Write your unit tests in the `tests` directory.
- Check if tests are working via `pytest`.
- Check for type errors via `mypy`.
- Check for style errors via `flake8`.
- Check for code quality via `pylint`.

### 8. Write descriptive commit messages

Try to keep individual commits easy to understand (changing dozens of files, writing 100's of lines of code is not!).

```bash
git commit -m '#42: Add useful new feature that does this.'
```

### 9. Push changes in the branch to your forked repository on GitHub

```bash
git push origin feature-xyz
```

Make sure to check if the CI pipeline is successful.

### 10. Create a merge request
