#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cubin-or-executable>" >&2
  exit 2
fi

command -v nvdisasm >/dev/null || { echo "nvdisasm is not on PATH" >&2; exit 1; }
nvdisasm --print-line-info --print-code "$1"

