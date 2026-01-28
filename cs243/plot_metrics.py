#!/usr/bin/env python3
"""Generate visualization plots for KV cache sparsification benchmark results.

This script creates comparative bar charts showing performance metrics across
different KV cache sparsification strategies (no-op, free-block, sparse-copy, spvllm)
at various token budgets.

Usage:
    python plot_metrics.py  # Generates plots in cs243/plots/
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set consistent plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# Directory constants
CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "results"
PLOTS_DIR = CURRENT_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# Strategy configurations
INTERNAL_STRATEGIES = ["no-op", "free-block", "sparse-copy", "spvllm"]
STRATEGY_COLORS = {
    "no-op": "#1f77b4",      # Blue
    "free-block": "#ff7f0e",  # Orange
    "sparse-copy": "#2ca02c", # Green
    "spvllm": "#d62728",      # Red
}
BUDGETS = [256, 512, 1024]
BATCH_SIZES = [2048]

# Plot configuration
BAR_WIDTH = 0.2
BAR_ALPHA = 0.7


@dataclass
class MetricConfig:
    """Configuration for a metric to plot."""
    key: str
    label: str
    formatter: str = "{:.1f}"
    is_percentage: bool = False
    use_scientific: bool = False

    def format_value(self, value: float) -> str:
        """Format a value for display on the plot."""
        if self.use_scientific:
            return f"{value:.1e}"
        if self.is_percentage:
            return f"{value:.1%}"
        return self.formatter.format(value)


# Define all metrics to plot
METRICS = [
    # Timing metrics
    MetricConfig("raw_duration", "Time taken (s)"),
    MetricConfig("raw_total_input_tokens", "Total input tokens", use_scientific=True),
    MetricConfig("raw_total_output_tokens", "Total output tokens", use_scientific=True),

    # Latency metrics (raw)
    MetricConfig("raw_p99_ttft_ms", "P99 TTFT (ms)"),
    MetricConfig("raw_p99_tpot_ms", "P99 TPOT (ms)"),
    MetricConfig("raw_p99_itl_ms", "P99 ITL (ms)"),

    # Throughput metrics (raw)
    MetricConfig("raw_request_throughput", "Request throughput (req/s)"),
    MetricConfig("raw_total_token_throughput", "Total token throughput (token/s)"),

    # Effective metrics (normalized by GPU utilization)
    MetricConfig("eff_request_throughput", "Effective request throughput (req/s)"),
    MetricConfig("eff_total_token_throughput", "Effective token throughput (token/s)"),
    MetricConfig("eff_p99_tpot_ms", "Effective P99 TPOT (ms)"),
    MetricConfig("eff_p99_itl_ms", "Effective P99 ITL (ms)"),

    # System metrics
    MetricConfig("num_batched_tokens_mean", "Mean batch size"),
    MetricConfig("num_preempted_total", "Total preempted requests", formatter="{:d}"),
    MetricConfig("attn_total_time_ms", "Attention forward time (ms)", use_scientific=True),

    # GPU utilization
    MetricConfig("gpu_utilization_mean", "Mean GPU utilization", is_percentage=True),
    MetricConfig("gpu_utilization_p90", "P90 GPU utilization", is_percentage=True),

    # Fragmentation metrics
    MetricConfig("frag_ratio_p99", "P99 Internal fragmentation", is_percentage=True),
    MetricConfig("frag_ratio_mean", "Mean Internal fragmentation", is_percentage=True),

    # Copy overhead
    MetricConfig("copy_total_time_ms", "Sparse copy overhead (ms)", use_scientific=True),
]


def build_experiment_key(batch_size: int, budget: int, internal: str) -> str:
    """Build the key for looking up experiment results."""
    budget_str = "max" if budget == "max" else str(budget)
    return f"{batch_size}-sharegpt-sharegpt.json-h2o-{budget_str}-1-{internal}"


def get_baseline_key(batch_size: int) -> str:
    """Get the key for the baseline (no sparsification) experiment."""
    return build_experiment_key(batch_size, "max", "no-op")


def extract_metric_values(
    data: Dict[str, Any],
    metric_key: str,
    batch_size: int,
    budgets: List[int],
    strategies: List[str]
) -> Dict[int, List[float]]:
    """Extract metric values for all budgets and strategies.

    Returns:
        Dict mapping budget -> list of values for each strategy.
    """
    result = {}
    for budget in budgets:
        values = []
        for strategy in strategies:
            key = build_experiment_key(batch_size, budget, strategy)
            if key in data and metric_key in data[key]:
                values.append(data[key][metric_key])
            else:
                values.append(0.0)
        result[budget] = values
    return result


def plot_metric(
    data: Dict[str, Any],
    metric: MetricConfig,
    batch_size: int,
    output_dir: Path,
) -> Optional[str]:
    """Create a bar chart comparing strategies across budgets for a metric.

    Returns:
        Filename of the generated plot, or None if plot could not be created.
    """
    # Extract data
    values_by_budget = extract_metric_values(
        data, metric.key, batch_size, BUDGETS, INTERNAL_STRATEGIES
    )

    # Check if we have any data
    if all(all(v == 0 for v in vals) for vals in values_by_budget.values()):
        return None

    values = np.array([values_by_budget[b] for b in BUDGETS])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each strategy
    x = np.arange(len(BUDGETS))
    offset = (len(INTERNAL_STRATEGIES) - 1) * BAR_WIDTH / 2

    for i, strategy in enumerate(INTERNAL_STRATEGIES):
        positions = x + i * BAR_WIDTH - offset
        color = STRATEGY_COLORS.get(strategy, f"C{i}")

        bars = ax.bar(
            positions,
            values[:, i],
            BAR_WIDTH,
            alpha=BAR_ALPHA,
            edgecolor="black",
            linewidth=0.5,
            label=strategy,
            color=color,
        )

        # Add value labels on bars
        for j, (pos, val) in enumerate(zip(positions, values[:, i])):
            if val > 0:
                ax.text(
                    pos,
                    val * 1.01,
                    metric.format_value(val),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    # Add baseline reference line
    baseline_key = get_baseline_key(batch_size)
    if baseline_key in data and metric.key in data[baseline_key]:
        baseline_value = data[baseline_key][metric.key]
        ax.axhline(
            baseline_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="vllm (no sparsification)",
            alpha=0.7,
        )

    # Configure axes
    ax.set_xlabel("Budget (tokens)", fontsize=11)
    ax.set_ylabel(metric.label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS])

    if metric.is_percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    # Add legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(INTERNAL_STRATEGIES) + 1,
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add title
    ax.set_title(f"{metric.label} by Strategy and Budget", fontsize=12, pad=30)

    plt.tight_layout()

    # Save figure
    filename = f"metric-{batch_size}-{metric.key.replace('/', '_')}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filename


def generate_summary_plot(data: Dict[str, Any], batch_size: int, output_dir: Path) -> None:
    """Generate a summary comparison plot with key metrics."""
    key_metrics = [
        ("raw_request_throughput", "Throughput (req/s)"),
        ("frag_ratio_p99", "P99 Fragmentation"),
        ("raw_p99_tpot_ms", "P99 TPOT (ms)"),
        ("num_preempted_total", "Preemptions"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric_key, label) in zip(axes, key_metrics):
        values_by_budget = extract_metric_values(
            data, metric_key, batch_size, BUDGETS, INTERNAL_STRATEGIES
        )
        values = np.array([values_by_budget[b] for b in BUDGETS])

        x = np.arange(len(BUDGETS))
        offset = (len(INTERNAL_STRATEGIES) - 1) * BAR_WIDTH / 2

        for i, strategy in enumerate(INTERNAL_STRATEGIES):
            positions = x + i * BAR_WIDTH - offset
            color = STRATEGY_COLORS.get(strategy, f"C{i}")
            ax.bar(
                positions, values[:, i], BAR_WIDTH,
                alpha=BAR_ALPHA, edgecolor="black", linewidth=0.5,
                label=strategy, color=color
            )

        ax.set_xlabel("Budget (tokens)")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_title(label)

        if "Fragmentation" in label:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle(f"KV Cache Sparsification Comparison (Batch Size: {batch_size})",
                 fontsize=14, y=1.05)
    plt.tight_layout()

    filepath = output_dir / f"summary-{batch_size}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated summary plot: {filepath.name}")


def main() -> None:
    """Main entry point for plot generation."""
    print("\033[34;1m" + "=" * 60 + "\033[0m")
    print("\033[34;1mGenerating KV Cache Sparsification Plots\033[0m")
    print("\033[34;1m" + "=" * 60 + "\033[0m\n")

    # Load analysis results
    results_path = RESULTS_DIR / "analyze.json"
    if not results_path.exists():
        print(f"Error: Analysis results not found at {results_path}")
        print("Run analyze.py first to generate results.")
        return

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} experiment results\n")

    # Generate plots for each batch size
    for batch_size in BATCH_SIZES:
        print(f"Generating plots for batch_size={batch_size}...")

        # Generate individual metric plots
        generated = 0
        for metric in METRICS:
            filename = plot_metric(data, metric, batch_size, PLOTS_DIR)
            if filename:
                print(f"  Generated: {filename}")
                generated += 1

        # Generate summary plot
        generate_summary_plot(data, batch_size, PLOTS_DIR)

        print(f"  Total plots generated: {generated + 1}\n")

    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
