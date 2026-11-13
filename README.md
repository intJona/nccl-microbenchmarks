# NCCL All-Reduce Sweep

Two-GPU NCCL all-reduce sweep for comparing PCIe configurations (topology, lane width,
ACS, switch vs. direct attach, ...). It is a thin wrapper around NVIDIA's
[nccl-tests](https://github.com/NVIDIA/nccl-tests) `all_reduce_perf`.

## Why a wrapper?

Running `all_reduce_perf` by hand works, but comparing PCIe configurations needs every run to be
identical and traceable. The scripts handle that:

- **Fixed parameters.** One default sweep (8 B – 256 MiB, same warm-up and iteration counts) is
  used for every configuration, so differences come from the hardware, not the flags.
- **Captured environment.** Each run stores the command, GPU model/driver/power/clocks and the
  `nvidia-smi topo -m` topology next to its output.
- **Labelled, repeatable runs.** `CONFIG_NAME` names the physical setup and `RUNS` repeats it,
  one timestamped folder per run.
- **Build helper.** `build.sh` fetches a pinned `nccl-tests` release and points it at the NCCL
  already installed in your Python environment, instead of requiring a separate NCCL install.
- **Shared results location.** Results can be sent to a shared directory (see
  [Storage](#storage)).

On GPUs without P2P support, NCCL falls back to another transport and the numbers reflect that.
Run with `NCCL_DEBUG=INFO` to see which path was chosen.

## Requirements

- Linux, 2 NVIDIA GPUs, `nvidia-smi`
- CUDA toolkit with `nvcc` (set `CUDA_TOOLKIT_ROOT` if it is not at the script's default location)
- A conda env named `vllm` whose site-packages contain NCCL (`nvidia/nccl`: `nccl.h` and `libnccl.so.2`)
- `git`, `make`, and internet access for the first build (set `https_proxy` if you are behind a proxy)

`scripts/activate.sh` hard-codes the env name (`vllm`) and the Python version in the
site-packages path. Edit it to match your system.

## Build (once)

```bash
conda activate vllm
./scripts/build.sh
```

This clones `nccl-tests` (default `v2.16.9`; override with `NCCL_TESTS_REF=<tag>`), links the
env's NCCL into `~/.cache/nccl-tests/`, and builds under `src/nccl-tests/build/`.

## Run a sweep

```bash
CONFIG_NAME=my_config CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_full_sweep.sh
```

| Variable / argument | Default | Description |
|---|---|---|
| `CONFIG_NAME` | `unnamed` | Label for the physical configuration, used in the result folder name |
| `CUDA_VISIBLE_DEVICES` | all GPUs | Which GPUs to test |
| `RUNS` | `1` | Repeat the sweep back-to-back; each run gets its own folder |
| extra arguments | `-b 8 -e 256M -f 2 -g 2 -w 5 -n 20` | Replace the default `all_reduce_perf` flags |

Example, three repeats with the defaults:

```bash
RUNS=3 CONFIG_NAME=my_config CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_full_sweep.sh
```

Each run writes `results/<timestamp>_full_sweep_<CONFIG_NAME>/` containing:

- `environment.txt` — command, GPU info, topology
- `all_reduce_8B_to_256MiB.txt` — raw `all_reduce_perf` output

## Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The dashboard only needs `streamlit` and `plotly`, not the NCCL/CUDA environment.

## Storage

Results location, highest priority first:

1. `NCCL_BENCH_RESULTS_DIR` environment variable
2. `.storage-config` (gitignored; copy from `.storage-config.example`)
3. `./results`
