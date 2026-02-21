#!/bin/bash
set -euo pipefail

# Keep a single implementation in project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec bash "$ROOT_DIR/post_process_all.sh" "$@"
