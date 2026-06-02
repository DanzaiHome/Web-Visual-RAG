#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${SCRIPT_DIR}/cache"

if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "cache directory does not exist: ${CACHE_DIR}"
  exit 0
fi

find "${CACHE_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
echo "cleared cache contents: ${CACHE_DIR}"
