#!/usr/bin/env python3
"""Generate motivation plot showing throughput improvement with sparsification."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "results"
PLOTS_DIR = CURRENT_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

BATCH_SIZES = [2048]
METRIC_KEY = "eff_request_throughput"  # Updated key name


def main() -> None:
    """Generate motivation comparison plot."""
    print("\033[34;1mGenerating Motivation Plot\033[0m\n")

    results_path = RESULTS_DIR / "analyze.json"
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        return

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for batch_size in BATCH_SIZES:
        # Get baseline throughput (no sparsification)
        baseline_key = f"{batch_size}-sharegpt-sharegpt.json-h2o-max-1-no-op"
        if baseline_key not in data or METRIC_KEY not in data[baseline_key]:
            print(f"Warning: No baseline data for batch_size={batch_size}")
            continue

        baseline_throughput = data[baseline_key][METRIC_KEY]

        # Find maximum throughput with sparsification
        max_throughput = baseline_throughput
        for key, value in data.items():
            if key.startswith(f"{batch_size}-") and key != baseline_key:
                if METRIC_KEY in value:
                    max_throughput = max(max_throughput, value[METRIC_KEY])

        speedup = max_throughput / baseline_throughput

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        bars = ax.bar(
            ["Without\nSparsification", "With\nSparsification"],
            [baseline_throughput, max_throughput],
            color=["#4878CF", "#6ACC65"],
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
        )

        # Add speedup labels
        ax.text(0, baseline_throughput + max_throughput * 0.02, "1x",
                ha="center", va="bottom", fontweight="bold", fontsize=16)
        ax.text(1, max_throughput + max_throughput * 0.02, f"{speedup:.2f}x",
                ha="center", va="bottom", fontweight="bold", fontsize=16)

        # Add arrow showing improvement
        ax.annotate(
            "",
            xy=(1, max_throughput * 0.95),
            xytext=(0.15, baseline_throughput),
            arrowprops={
                "arrowstyle": "->",
                "lw": 3,
                "alpha": 0.6,
                "color": "#333333",
                "connectionstyle": "arc3,rad=0.2",
            },
        )

        ax.set_ylabel("Request Throughput (req/s)", fontsize=12)
        ax.set_ylim(0, max_throughput * 1.25)
        ax.set_title("Throughput Improvement with KV Cache Sparsification",
                     fontsize=13, fontweight="bold")

        plt.tight_layout()

        filename = f"motivation-{batch_size}.png"
        fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated: {filename}")

    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
