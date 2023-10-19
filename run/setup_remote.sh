#!/usr/bin/env bash
echo -e "\n" "Current remotes:"
git remote -v

# remotes
BERLIN="https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm.git"
HILDESHEIM="https://software.ismll.uni-hildesheim.de/ISMLL-internal/time-series/tsdm.git"
GITHUB="https://github.com/randolf-scholz/tsdm.git"

echo -e "\nDeleting all remotes..."
for remote_name in $(git remote); do
    git remote remove "${remote_name}"
done

echo -e "\nAdding remote ${BERLIN}..."
git remote add berlin $BERLIN
git remote set-url --add --push berlin $BERLIN
git remote set-url --add --push berlin $GITHUB
git remote set-url --add --push berlin $HILDESHEIM

echo -e "\nAdding remote ${GITHUB}..."
git remote add github $GITHUB
git remote set-url --add --push github $GITHUB
git remote set-url --add --push github $BERLIN
git remote set-url --add --push github $HILDESHEIM

echo -e "\nAdding remote ${HILDESHEIM}..."
git remote add hildesheim $HILDESHEIM
git remote set-url --add --push hildesheim $HILDESHEIM
git remote set-url --add --push hildesheim $BERLIN
git remote set-url --add --push hildesheim $GITHUB

echo -e "\nSetting default remote:"
git fetch berlin
git branch --set-upstream-to=berlin/main  main
git push -u berlin --all

echo -e "\nNew remote config:"
git remote -v
