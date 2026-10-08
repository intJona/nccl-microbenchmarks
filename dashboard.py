#!/usr/bin/env python3
"""Interactive comparison dashboard for saved NCCL all-reduce sweeps."""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent


def resolve_results_root() -> Path:
    """Results dir: $NCCL_BENCH_RESULTS_DIR, then .storage-config, then ./results."""
    env_override = os.environ.get("NCCL_BENCH_RESULTS_DIR")
    if env_override:
        return Path(env_override).expanduser()
    config_file = REPO_ROOT / ".storage-config"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                return Path(line).expanduser()
    return REPO_ROOT / "results"


RESULTS_ROOT = resolve_results_root()
RESULT_FILE_GLOB = "all_reduce_*.txt"
DATA_LINE = re.compile(
    r"^\s*(?P<size>\d+)\s+\d+\s+\w+\s+\w+\s+-?\d+\s+"
    r"(?P<time>[\d.]+)\s+(?P<algbw>[\d.]+)\s+(?P<busbw>[\d.]+)",
    re.MULTILINE,
)
# Okabe-Ito palette: distinguishable under deuteranopia/protanopia/tritanopia.
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
MARKER_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "star", "triangle-down", "pentagon"]


def read_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    except OSError:
        pass
    return values


def parse_result(path: Path) -> list[dict[str, float]]:
    rows = []
    try:
        content = path.read_text(errors="replace")
    except OSError:
        return rows
    for match in DATA_LINE.finditer(content):
        parsed = {key: float(match.group(key)) for key in ("size", "time", "algbw", "busbw")}
        rows.append(parsed)
    return rows


@st.cache_data(show_spinner=False)
def discover_runs() -> list[dict]:
    runs = []
    if not RESULTS_ROOT.exists():
        return runs
    for result_file in sorted(RESULTS_ROOT.glob(f"*/{RESULT_FILE_GLOB}")):
        rows = parse_result(result_file)
        if not rows:
            continue
        run_dir = result_file.parent
        environment = read_environment(run_dir / "environment.txt")
        config = environment.get("config_name")
        if not config:
            config = run_dir.name.split("_full_sweep_")[-1] if "_full_sweep_" in run_dir.name else "unlabelled"
        runs.append({
            "id": run_dir.name,
            "config": config,
            "rows": rows,
            "environment": environment,
            "file": result_file.name,
        })
    return runs


@st.cache_data(show_spinner=False)
def discover_skipped_results() -> list[str]:
    skipped = []
    if not RESULTS_ROOT.exists():
        return skipped
    for result_file in sorted(RESULTS_ROOT.glob(f"*/{RESULT_FILE_GLOB}")):
        if not parse_result(result_file):
            skipped.append(str(result_file))
    return skipped


def aggregate_runs(runs: list[dict]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, dict[float, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for row in run["rows"]:
            grouped[run["config"]][row["size"]].append(row)
    return {
        config: [
            {"size": size, **{metric: median(row[metric] for row in rows) for metric in ("time", "algbw", "busbw")}}
            for size, rows in sorted(by_size.items())
        ]
        for config, by_size in grouped.items()
    }


def chart(title: str, metric: str, unit: str, runs: list[dict], aggregated: dict[str, list[dict[str, float]]]) -> go.Figure:
    figure = go.Figure()
    styles = {
        config: {
            "color": PALETTE[index % len(PALETTE)],
            "dash": DASH_STYLES[index % len(DASH_STYLES)],
            "symbol": MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)],
        }
        for index, config in enumerate(sorted(aggregated))
    }
    for run in runs:
        style = styles[run["config"]]
        figure.add_trace(go.Scatter(
            x=[row["size"] for row in run["rows"]], y=[row[metric] for row in run["rows"]],
            mode="lines", line={"color": style["color"], "width": 1, "dash": "dot"}, opacity=0.28,
            hovertemplate=f"{run['config']} | {run['id']}<br>%{{x:,.0f}} B<br>%{{y:.3f}} {unit}<extra></extra>",
            showlegend=False,
        ))
    for config, rows in aggregated.items():
        style = styles[config]
        figure.add_trace(go.Scatter(
            x=[row["size"] for row in rows], y=[row[metric] for row in rows], mode="lines+markers",
            name=f"{config} median",
            line={"color": style["color"], "width": 3, "dash": style["dash"]},
            marker={"size": 8, "symbol": style["symbol"], "line": {"width": 1, "color": "white"}},
            hovertemplate=f"{config} median<br>%{{x:,.0f}} B<br>%{{y:.3f}} {unit}<extra></extra>",
        ))
    figure.update_layout(title=title, height=430, margin={"l": 8, "r": 8, "t": 46, "b": 92},
                         legend={"orientation": "h", "y": -0.24, "yanchor": "top"}, template="plotly_white")
    figure.update_xaxes(title="Message size (bytes)", type="log", tickformat="~s")
    figure.update_yaxes(title=f"{metric.title()} ({unit})", rangemode="tozero")
    return figure


def decade_sizes(sizes) -> list[float]:
    """Smallest size recorded in each log10 decade, plus the overall max -- regular sample points for a
    log-scale sweep. The max is forced in because it can share a decade with (and get shadowed by) a
    smaller size, e.g. 256 MB and 128 MB both fall in the 10^8-10^9 decade."""
    sizes = [size for size in sizes if size > 0]
    if not sizes:
        return []
    buckets: dict[int, float] = {}
    for size in sizes:
        decade = math.floor(math.log10(size))
        if decade not in buckets or size < buckets[decade]:
            buckets[decade] = size
    return sorted(set(buckets.values()) | {max(sizes)})


def format_size(size: float) -> str:
    if size >= 1024 * 1024 and size % (1024 * 1024) == 0:
        return f"{int(size // (1024 * 1024))} MB"
    if size >= 1024 and size % 1024 == 0:
        return f"{int(size // 1024)} KB"
    return f"{int(size)} B"


def decade_table(runs: list[dict], aggregated: dict[str, list[dict[str, float]]], metric: str) -> list[dict]:
    """One row per config with the metric value at the start of each log-scale decade."""
    all_sizes = {row["size"] for rows in aggregated.values() for row in rows}
    decades = decade_sizes(all_sizes)
    table = []
    for config in sorted(aggregated):
        matching_runs = [run for run in runs if run["config"] == config]
        environment = matching_runs[0]["environment"] if matching_runs else {}
        value_by_size = {row["size"]: row[metric] for row in aggregated[config]}
        entry = {
            "Configuration": config,
            "Runs": len(matching_runs),
            "GPU selection": environment.get("cuda_visible_devices", "not recorded"),
        }
        for size in decades:
            entry[format_size(size)] = value_by_size.get(size)
        table.append(entry)
    return table


def main() -> None:
    st.set_page_config(page_title="NCCL PCIe Results", page_icon="N", layout="wide")
    st.title("NCCL PCIe Microbenchmark Results")
    all_runs = discover_runs()
    skipped_results = discover_skipped_results()
    if not all_runs:
        st.info("No parseable NCCL result files found under results/.")
        return

    if skipped_results:
        st.warning("Skipped result files with no parseable NCCL data: " + ", ".join(skipped_results))

    configs = sorted({run["config"] for run in all_runs})
    selected_configs = st.sidebar.multiselect("Configurations", configs, default=configs)
    selected_runs = [run for run in all_runs if run["config"] in selected_configs]
    if not selected_runs:
        st.warning("Select at least one configuration.")
        return

    aggregated = aggregate_runs(selected_runs)
    st.caption(f"{len(selected_runs)} run(s) across {len(aggregated)} configuration(s). Thin lines are individual runs; solid lines are per-size medians.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart("All-reduce latency", "time", "us", selected_runs, aggregated), use_container_width=True)
    with right:
        st.plotly_chart(chart("All-reduce bus bandwidth", "busbw", "GB/s", selected_runs, aggregated), use_container_width=True)

    st.subheader("All-reduce latency by decade (us)")
    st.dataframe(decade_table(selected_runs, aggregated, "time"), use_container_width=True, hide_index=True)

    st.subheader("Bus bandwidth by decade (GB/s)")
    st.dataframe(decade_table(selected_runs, aggregated, "busbw"), use_container_width=True, hide_index=True)

    with st.expander("Captured run metadata"):
        for run in selected_runs:
            st.markdown(f"**{run['id']}** ({run['file']})")
            st.code("\n".join(f"{key}={value}" for key, value in run["environment"].items()) or "No environment metadata found.")


if __name__ == "__main__":
    main()