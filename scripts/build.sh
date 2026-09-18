#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/activate.sh

NCCL_TESTS_REF="${NCCL_TESTS_REF:-v2.16.9}"
NCCL_PYTHON_ROOT="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nccl"
VENDOR_ROOT="$NCCL_TESTS_NCCL_HOME"

mkdir -p "$VENDOR_ROOT/include" "$VENDOR_ROOT/lib"
ln -sfn "$NCCL_PYTHON_ROOT/include/nccl.h" "$VENDOR_ROOT/include/nccl.h"
ln -sfn "$NCCL_PYTHON_ROOT/lib/libnccl.so.2" "$VENDOR_ROOT/lib/libnccl.so"
ln -sfn "$NCCL_PYTHON_ROOT/lib/libnccl.so.2" "$VENDOR_ROOT/lib/libnccl.so.2"

if [[ ! -d src/nccl-tests/.git ]]; then
    echo "Fetching nccl-tests $NCCL_TESTS_REF..."
    git clone --depth 1 --branch "$NCCL_TESTS_REF" https://github.com/NVIDIA/nccl-tests.git src/nccl-tests
fi

export NCCL_HOME="$VENDOR_ROOT"
make -C src/nccl-tests/src clean
make -C src/nccl-tests/src -j"$(nproc)"
