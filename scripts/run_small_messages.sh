#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/activate.sh
source scripts/lib_storage.sh
RESULTS_ROOT="$(resolve_results_root)"

BINARY="$PROJECT_ROOT/src/nccl-tests/build/all_reduce_perf"
if [[ ! -x "$BINARY" ]]; then
    echo "Missing $BINARY. Run scripts/build.sh first." >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_ID"
mkdir -p "$OUTPUT_DIR"

{
    echo "timestamp_utc=$(date --utc +%FT%TZ)"
    echo "command=$BINARY -b 8 -e 1M -f 2 -g 2 -w 20 -n 100"
    nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
    nvidia-smi topo -m
} >"$OUTPUT_DIR/environment.txt"

NCCL_DEBUG="${NCCL_DEBUG:-WARN}" "$BINARY" -b 8 -e 1M -f 2 -g 2 -w 20 -n 100 | tee "$OUTPUT_DIR/all_reduce_8B_to_1MiB.txt"
