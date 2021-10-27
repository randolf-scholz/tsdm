for remote_name in $(git remote); do 
    git remote remove "${remote_name}"
done

HILDESHEIM="https://software.ismll.uni-hildesheim.de/ISMLL-internal/special-interest-group-time-series/tsdm.git"
BERLIN="https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/tsdm.git"

git remote add berlin $BERLIN
git remote set-url --add --push berlin $BERLIN
git remote set-url --add --push berlin $HILDESHEIM

git remote add hildesheim $HILDESHEIM
git remote set-url --add --push hildesheim $BERLIN
git remote set-url --add --push hildesheim $HILDESHEIM

git remote -v

git fetch berlin
git branch --set-upstream-to=berlin/main  main


