#!/usr/bin/env bash
# Initialize git + DVC for regression_health on your local machine.
# (The project was assembled in a sandbox whose filesystem can't host a git repo,
#  so a partial `.git/` may exist — this script removes and recreates it cleanly.)
set -euo pipefail
cd "$(dirname "$0")"

# 0. clean any partial repo created in the build environment
rm -rf .git

# 1. git
git init
git add -A                       # raw + derived data are excluded via .gitignore
git commit -m "Initial commit: regression_health (profile, pipeline, DVC pointer)"

# 2. DVC: point the remote at real storage, then push the raw file
#    edit the URL first (local dir, s3://, gdrive://, ...)
# dvc remote modify storage url /abs/path/to/dvc-storage
dvc add export.csv               # recomputes hash, populates local DVC cache
git add export.csv.dvc .gitignore
git commit -m "Track raw export.csv with DVC"
# dvc push                        # uploads bytes to the configured remote

echo "Done. 'git status' should be clean; 'dvc status' should say up to date."
