"""Plot generation for context scaling characterization results.

Reads JSONL files from a results directory and produces:
  1. tok/s vs context fill
  2. Memory vs context fill
  3. Thinking ratio vs context fill
  4. Time per turn vs context fill
  5. Combined 2x2 dashboard

Usage as library::

    from aeo_quant.plots.context_scaling import generate_dashboard
    generate_dashboard("results/context_scaling", output_dir="results/plots")
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

# One color per target for visual consistency across charts
TARGET_COLORS = {
    16384: "#1f77b4",
    32768: "#ff7f0e",
    49152: "#2ca02c",
    65536: "#d62728",
    131072: "#9467bd",
}


def load_run(results_dir: Path, target: int) -> list[dict]:
    """Load successful turn records from a run's JSONL file."""
    jsonl_path = results_dir / f"run_{target}.jsonl"
    if not jsonl_path.exists():
        return []
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Skip error-only records (no metrics)
            if rec.get("error") and "n_input_tokens" not in rec:
                continue
            records.append(rec)
    return records


def label_for_target(target: int) -> str:
    """Format a context target as a human-readable label (e.g. 16384 -> '16K')."""
    if target >= 1024:
        return f"{target // 1024}K"
    return str(target)


def plot_tok_per_s(ax, all_runs: dict[int, list[dict]]) -> None:
    """Plot decode speed vs context fill."""
    ax.set_title("Decode Speed vs Context Fill")
    ax.set_xlabel("Context Tokens")
    ax.set_ylabel("tok/s")
    for target, records in sorted(all_runs.items()):
        if not records:
            continue
        x = [r["n_input_tokens"] for r in records]
        y = [r["tok_per_s"] for r in records]
        color = TARGET_COLORS.get(target, "gray")
        ax.plot(x, y, "o-", label=label_for_target(target), color=color, markersize=4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_memory(ax, all_runs: dict[int, list[dict]]) -> None:
    """Plot memory usage vs context fill."""
    ax.set_title("Memory Usage vs Context Fill")
    ax.set_xlabel("Context Tokens")
    ax.set_ylabel("System Used (GB)")
    for target, records in sorted(all_runs.items()):
        if not records:
            continue
        x = [r["n_input_tokens"] for r in records]
        y = [r["sys_used_after_gb"] for r in records]
        color = TARGET_COLORS.get(target, "gray")
        ax.plot(x, y, "o-", label=label_for_target(target), color=color, markersize=4)
    # 90 GB cap line
    ax.axhline(y=90.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="90 GB cap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_thinking_ratio(ax, all_runs: dict[int, list[dict]]) -> None:
    """Plot thinking ratio vs context fill."""
    ax.set_title("Thinking Ratio vs Context Fill")
    ax.set_xlabel("Context Tokens")
    ax.set_ylabel("Thinking / Total Generated")
    for target, records in sorted(all_runs.items()):
        if not records:
            continue
        x = [r["n_input_tokens"] for r in records]
        y = [r["thinking_ratio"] for r in records]
        color = TARGET_COLORS.get(target, "gray")
        ax.plot(x, y, "o-", label=label_for_target(target), color=color, markersize=4)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_time_per_turn(ax, all_runs: dict[int, list[dict]]) -> None:
    """Plot time per turn vs context fill."""
    ax.set_title("Time per Turn vs Context Fill")
    ax.set_xlabel("Context Tokens")
    ax.set_ylabel("Total Time (seconds)")
    for target, records in sorted(all_runs.items()):
        if not records:
            continue
        x = [r["n_input_tokens"] for r in records]
        y = [r["total_time_s"] for r in records]
        color = TARGET_COLORS.get(target, "gray")
        ax.plot(x, y, "o-", label=label_for_target(target), color=color, markersize=4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def generate_dashboard(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    title: str = "Context Scaling",
) -> Path:
    """Generate individual plots and a combined 2x2 dashboard.

    Args:
        results_dir: Directory containing ``run_*.jsonl`` files.
        output_dir: Where to save plots. Defaults to ``results_dir / "plots"``.
        title: Title for the combined dashboard figure.

    Returns:
        Path to the output directory containing the generated plots.

    Raises:
        FileNotFoundError: If *results_dir* does not exist or contains no data.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results directory not found: {results_dir}")

    if output_dir is None:
        output_dir = results_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available run files
    run_files = sorted(results_dir.glob("run_*.jsonl"))
    if not run_files:
        raise FileNotFoundError(f"no run_*.jsonl files found in {results_dir}")

    all_runs: dict[int, list[dict]] = {}
    for f in run_files:
        # Extract target from filename: run_16384.jsonl -> 16384
        target = int(f.stem.split("_", 1)[1])
        records = load_run(results_dir, target)
        if records:
            all_runs[target] = records
            print(f"[load] target={target:>7,}: {len(records)} turns")
        else:
            print(f"[load] target={target:>7,}: no data (skipping)")

    if not all_runs:
        raise FileNotFoundError(f"no valid data in any run file under {results_dir}")

    # Individual plots
    plot_funcs = [
        ("tok_per_s_vs_context", plot_tok_per_s),
        ("memory_vs_context", plot_memory),
        ("thinking_ratio_vs_context", plot_thinking_ratio),
        ("time_per_turn_vs_context", plot_time_per_turn),
    ]

    for name, func in plot_funcs:
        fig, ax = plt.subplots(figsize=(10, 6))
        func(ax, all_runs)
        fig.tight_layout()
        out = output_dir / f"{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] {out}")

    # Combined 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{title} Dashboard", fontsize=14, y=0.98)

    plot_tok_per_s(axes[0, 0], all_runs)
    plot_memory(axes[0, 1], all_runs)
    plot_thinking_ratio(axes[1, 0], all_runs)
    plot_time_per_turn(axes[1, 1], all_runs)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = output_dir / "combined_dashboard.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")

    print(f"\n[done] all plots saved to {output_dir}")
    return output_dir
