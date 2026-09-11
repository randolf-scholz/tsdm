#!/usr/bin/env -S just --justfile
mod docs
mod tests

export GIT_ROOT := `git rev-parse --show-toplevel`
export GIT_PREFIX := `git rev-parse --show-prefix`
export TEST_DIR := GIT_ROOT / "tests"

[default]
[doc('List available commands.')]
help:
    @just --justfile {{ justfile() }} --list --list-submodules --unsorted

[doc('Remove generated Python cache and build artifacts.')]
clean:
    uv run pyclean {{ GIT_ROOT }} --debris

[doc('Show ignored files below the target directory.')]
show-ignored target=TEST_DIR:
    #!/usr/bin/env bash
    git -C "{{ target }}" ls-files -z --others -i --exclude-standard \
      --directory -- |
    while IFS= read -r -d '' p; do
      printf '%s\n' "${p#$prefix}"
    done

[doc('Reconfigure the repository Git remotes.')]
setup-remote:
    #!/usr/bin/env bash
    echo -e "\n" "Current remotes:"
    git remote -v

    # remotes
    #BERLIN="https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm.git"
    HILDESHEIM="https://software.ismll.uni-hildesheim.de/ISMLL-internal/time-series/tsdm.git"
    GITHUB="https://github.com/randolf-scholz/tsdm.git"

    echo -e "\nDeleting all remotes..."
    for remote_name in $(git remote); do
        git remote remove "${remote_name}"
    done

    #echo -e "\nAdding remote ${BERLIN}..."
    #git remote add berlin $BERLIN
    #git remote set-url --add --push berlin $BERLIN
    #git remote set-url --add --push berlin $GITHUB
    #git remote set-url --add --push berlin $HILDESHEIM

    echo -e "\nAdding remote ${GITHUB}..."
    git remote add github $GITHUB
    git remote set-url --add --push github $GITHUB
    #git remote set-url --add --push github $BERLIN
    git remote set-url --add --push github $HILDESHEIM

    echo -e "\nAdding remote ${HILDESHEIM}..."
    git remote add hildesheim $HILDESHEIM
    git remote set-url --add --push hildesheim $HILDESHEIM
    #git remote set-url --add --push hildesheim $BERLIN
    git remote set-url --add --push hildesheim $GITHUB

    echo -e "\nSetting default remote:"
    git fetch github
    git branch --set-upstream-to=github/main  main
    git push -u github --all

    echo -e "\nNew remote config:"
    git remote -v

[doc('Upgrade dependencies and run project checks.')]
upgrade:
    uv sync --upgrade
    git add .
    uv run prek run -a
    git add .
    uv run prek run -a
