# PCIe NCCL Microbenchmarks

Reproducible two-GPU NCCL microbenchmarks for this workstation. The two RTX 3090 GPUs connect as `NODE` in `nvidia-smi topo -m`, so these runs measure PCIe peer-to-peer collective behavior plus NCCL launch and synchronization overhead, rather than NVLink throughput.

This project intentionally reuses the existing `vllm` Conda environment for NCCL 2.28.9 development headers/runtime and uses the installed CUDA 13.3 SDK for compilation. A second environment would duplicate those large packages and risk mixing NCCL ABIs.

## First build

```bash
cd "/home/intel/PCIe NCCL Benchmarks"
conda activate vllm
./scripts/build.sh
```

`build.sh` downloads the official NVIDIA `nccl-tests` release `v2.16.9`, stages local NCCL-library symlinks under `~/.cache/nccl-tests/` (the upstream Makefiles cannot handle spaces in compiler paths), and builds the executables under `src/nccl-tests/build/`. Override the source tag with `NCCL_TESTS_REF=<tag> ./scripts/build.sh` when needed.

## Small-message all-reduce

```bash
./scripts/run_small_messages.sh
```

This executes `all_reduce_perf` over powers of two from 8 bytes through 1 MiB, using both GPUs, 20 warm-ups, and 100 measured iterations. Results and the exact topology/environment are stored in `results/<UTC timestamp>/`.

The output's `time` column is the latency per collective. `algbw` is useful payload bandwidth and `busbw` accounts for the all-reduce algorithm. For AI-serving-style launch/synchronization sensitivity, compare the 8 B through 64 KiB rows; the larger rows characterize the sustained PCIe collective path.

## Full-range comparison sweep

```bash
CONFIG_NAME=node_x16 CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_full_sweep.sh
```

This runs the same all-reduce configuration from 8 B through 256 MiB (`-b 8 -e 256M -f 2 -g 2 -w 5 -n 20`) for every topology or lane-width comparison. `CONFIG_NAME` identifies the physical configuration in the result directory; `CUDA_VISIBLE_DEVICES` chooses the physical GPU pair. Pass your own `all_reduce_perf` flags as arguments to override the defaults, e.g. `./scripts/run_full_sweep.sh -b 8 -e 256M -f 2 -g 2 -w 5 -n 20`.

Set `RUNS=<count>` to repeat the sweep back-to-back under the same `CONFIG_NAME`, each producing its own timestamped result directory (default: 1), e.g. `RUNS=3 CONFIG_NAME=node_x16 CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_full_sweep.sh`.

The internal warm-ups and iterations make a single sweep stable, but repeat the full script at least three times per physical configuration when comparing small differences. Keep GPU pair, clocks, power limits, driver, CUDA/NCCL version, and all NCCL environment variables unchanged between configurations.

## Results dashboard

```bash
conda activate vllm
streamlit run dashboard.py
```

The dashboard overlays individual runs and per-size medians for latency and bus bandwidth, then summarizes representative message sizes for every selected configuration. Run every physical configuration at least three times with a distinct `CONFIG_NAME` before comparing medians.

## Storage location

Results are written to `results/` inside the repo by default, so a fresh clone works with no
setup. To point at a shared/network location instead (e.g. so results survive a re-clone), copy
`.storage-config.example` to `.storage-config` (gitignored) and set the path there, or set the
`NCCL_BENCH_RESULTS_DIR` environment variable. Precedence: env var > `.storage-config` > `results/`.

## Corporate proxy

Use the corporate HTTPS proxy when a build needs to fetch the pinned source:

```bash
export https_proxy=http://proxy-dmz.intel.com:912
export HTTPS_PROXY=http://proxy-dmz.intel.com:912
./scripts/build.sh
```