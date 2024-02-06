#!/usr/bin/env bash
set -e  # exit on first error

# navigate to the root of the project
cd "$(git rev-parse --show-toplevel)"
echo "$PWD"

# update dependencies
pre-commit autoupdate
# run pdm update and capture the output
output=$(pdm update --update-all | tee /dev/tty)

# create a commit message with everything after "🔒 Lock successful"
pattern="🔒 Lock successful"
commit_message=${output#"*$pattern"}
# error if the commit message is empty
if [ -z "$commit_message" ]; then
  echo "No changes detected. Exiting..."
  exit 1
fi

# commit the changes
set +e
pre-commit run pyproject-update-deps -a --hook-stage manual
git add .
git commit -m "chore(deps): update dependencies\n $commit_message"
