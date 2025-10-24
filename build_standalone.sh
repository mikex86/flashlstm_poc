#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
  echo "+ $*"
  "$@"
}

run python3 "${ROOT}/scripts/ensure_cutlass.py"
run /usr/bin/cmake -DCMAKE_BUILD_TYPE=Release -S "${ROOT}" -B "${ROOT}/build"
run /usr/bin/cmake --build "${ROOT}/build" --target run_lstm
