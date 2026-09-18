#!/usr/bin/env bash
# Source this file from the vLLM Conda environment before building or running.

if [[ "${CONDA_DEFAULT_ENV:-}" != "vllm" ]]; then
    echo "Activate the existing NCCL-compatible environment first: conda activate vllm" >&2
    return 1 2>/dev/null || exit 1
fi

NCCL_PYTHON_ROOT="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/nccl"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-13.3}"
if [[ ! -x "$CUDA_TOOLKIT_ROOT/bin/nvcc" || ! -f "$CUDA_TOOLKIT_ROOT/targets/x86_64-linux/include/cuda_runtime.h" || ! -f "$NCCL_PYTHON_ROOT/include/nccl.h" || ! -f "$NCCL_PYTHON_ROOT/lib/libnccl.so.2" ]]; then
    echo "Missing the CUDA 13.3 SDK or NCCL development artifacts required by nccl-tests." >&2
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export NCCL_TESTS_NCCL_HOME="${NCCL_TESTS_NCCL_HOME:-$HOME/.cache/nccl-tests/nccl}"
export LD_LIBRARY_PATH="$NCCL_TESTS_NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"
