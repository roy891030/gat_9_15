#!/usr/bin/env bash
set -euo pipefail

# Sync repository with its remote tracking branch to avoid stale changes.
# Usage: ./sync_latest.sh [remote] [branch]
# Defaults: remote=origin, branch=main

remote="${1:-origin}"
branch="${2:-main}"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "[sync_latest] Error: not inside a git repository." >&2
  exit 1
fi

if ! git remote get-url "$remote" >/dev/null 2>&1; then
  echo "[sync_latest] No remote named '$remote' configured." >&2
  echo "Add one with: git remote add $remote <REMOTE_URL>" >&2
  exit 1
fi

echo "[sync_latest] Fetching latest refs from '$remote'..."
git fetch "$remote"

echo "[sync_latest] Rebasing local branch onto '$remote/$branch'..."
git pull --rebase "$remote" "$branch"

echo "[sync_latest] Done. Current status:"
git status -sb
