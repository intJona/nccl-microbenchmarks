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

# Keep these fixed across PCIe configurations for comparable results.
DEFAULT_RUN_PARAMS=(-b 8 -e 256M -f 2 -g 2 -w 5 -n 20)
RUN_PARAMS=("$@")
if [[ ${#RUN_PARAMS[@]} -eq 0 ]]; then
    RUN_PARAMS=("${DEFAULT_RUN_PARAMS[@]}")
fi
CONFIG_NAME="${CONFIG_NAME:-unnamed}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-all}"
RUNS="${RUNS:-1}"
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
    echo "RUNS must be a positive integer, got '$RUNS'" >&2
    exit 1
fi

for ((run = 1; run <= RUNS; run++)); do
    RUN_ID="$(date +%Y%m%d-%H%M%S)_full_sweep_${CONFIG_NAME}"
    OUTPUT_DIR="$RESULTS_ROOT/$RUN_ID"
    mkdir -p "$OUTPUT_DIR"

    {
        echo "timestamp_utc=$(date --utc +%FT%TZ)"
        echo "config_name=$CONFIG_NAME"
        echo "cuda_visible_devices=$VISIBLE_GPUS"
        printf 'command=%q ' "$BINARY" "${RUN_PARAMS[@]}"
        printf '\n'
        nvidia-smi --query-gpu=name,driver_version,pci.bus_id,power.limit,clocks.max.graphics,clocks.max.mem --format=csv,noheader 2>&1
        nvidia-smi topo -m 2>&1
    } >"$OUTPUT_DIR/environment.txt"

    echo "Running full all-reduce sweep for configuration: $CONFIG_NAME (run $run/$RUNS)"
    NCCL_DEBUG="${NCCL_DEBUG:-WARN}" "$BINARY" "${RUN_PARAMS[@]}" 2>&1 | tee "$OUTPUT_DIR/all_reduce_8B_to_256MiB.txt"
    echo "Results saved in $OUTPUT_DIR"
done
