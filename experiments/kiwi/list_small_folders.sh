#!/bin/env bash

# recursively list all directories if their size is less than 10KB
# find . -type d -size -10k -exec du -sh {} \;


# recursively list all directories and their size with less than 10KB
find . -type d -print0 | while read -d $'\0' dir; do
    if [ $(du -s "$dir" | cut -f1) -lt 16 ]; then
        du -sh "$dir"
    fi
done

# ask for confirmation whether to delete the directories
read -p "Delete the above directories? [y/N] " -n 1 -r

# if yes, delete the directories
if [[ $REPLY =~ ^[Yy]$ ]]; then
    find . -type d -print0 | while read -d $'\0' dir; do
        if [ $(du -s "$dir" | cut -f1) -lt 16 ]; then
            rm -rf "$dir"
            echo "Deleted $dir"
        fi
    done
else # else, do nothing
    echo "Nothing deleted."
    exit 0
fi
