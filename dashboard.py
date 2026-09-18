#!/usr/bin/env python3
"""Interactive comparison dashboard for saved NCCL all-reduce sweeps."""

from __future__ import annotations

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
        # nccl-tests reports time in microseconds; keep dashboard latency in seconds.
        parsed["time"] = parsed["time"] / 1_000_000.0
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


def metric_at(rows: list[dict[str, float]], size: float, metric: str) -> float | None:
    return next((row[metric] for row in rows if row["size"] == size), None)


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
        st.plotly_chart(chart("All-reduce latency", "time", "s", selected_runs, aggregated), use_container_width=True)
    with right:
        st.plotly_chart(chart("All-reduce bus bandwidth", "busbw", "GB/s", selected_runs, aggregated), use_container_width=True)

    summary = []
    for config, rows in aggregated.items():
        matching_runs = [run for run in selected_runs if run["config"] == config]
        environment = matching_runs[0]["environment"]
        summary.append({
            "Configuration": config,
            "Runs": len(matching_runs),
            "8 B latency (s)": metric_at(rows, 8, "time"),
            "1 KiB latency (s)": metric_at(rows, 1024, "time"),
            "64 KiB latency (s)": metric_at(rows, 65536, "time"),
            "1 MiB busbw (GB/s)": metric_at(rows, 1048576, "busbw"),
            "256 MiB busbw (GB/s)": metric_at(rows, 268435456, "busbw"),
            "GPU selection": environment.get("cuda_visible_devices", "not recorded"),
        })
    st.subheader("Median Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("Captured run metadata"):
        for run in selected_runs:
            st.markdown(f"**{run['id']}** ({run['file']})")
            st.code("\n".join(f"{key}={value}" for key, value in run["environment"].items()) or "No environment metadata found.")


if __name__ == "__main__":
    main()