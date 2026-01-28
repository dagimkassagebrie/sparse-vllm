#!/usr/bin/env python3
"""Generate internal fragmentation time-series plots."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from numpy.typing import NDArray

sns.set_theme(style="whitegrid")

CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "results"
PLOTS_DIR = CURRENT_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# Configuration
BATCH_SIZES = [2048]
BUDGETS = [256, 512, 1024]
STRATEGIES = ["no-op", "free-block", "sparse-copy", "spvllm"]

STRATEGY_COLORS = {
    "no-op": "#1f77b4",
    "free-block": "#ff7f0e",
    "sparse-copy": "#2ca02c",
    "spvllm": "#d62728",
}


def load_fragmentation_data(
    batch_size: int,
    budget: int | str,
    internal: str
) -> Optional[NDArray]:
    """Load fragmentation data from numpy file."""
    filename = f"{batch_size}-sharegpt-sharegpt.json-h2o-{budget}-1-{internal}-frag.npy"
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        return None

    with filepath.open("rb") as f:
        num_active = np.load(f)
        num_total = np.load(f)

    return 1.0 - num_active / num_total


def plot_fragmentation_over_time(
    batch_size: int,
    budget: int,
    baseline: NDArray,
    strategy_data: Dict[str, NDArray]
) -> str:
    """Create fragmentation time-series plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline
    ax.plot(
        np.arange(len(baseline)),
        baseline,
        color="black",
        linestyle="--",
        linewidth=2,
        label="vllm (no sparsification)",
        alpha=0.7,
    )

    # Plot each strategy
    for strategy, frag_data in strategy_data.items():
        color = STRATEGY_COLORS.get(strategy, None)
        x = np.arange(len(frag_data))
        ax.plot(x, frag_data, label=strategy, color=color, linewidth=1.5)
        ax.fill_between(x, 0, frag_data, alpha=0.15, color=color)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Internal Fragmentation", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.set_title(f"Internal Fragmentation Over Time (Budget: {budget} tokens)", fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"frag-timeseries-{batch_size}-{budget}.png"
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filename


def main() -> None:
    """Generate all fragmentation plots."""
    print("\033[34;1mGenerating Fragmentation Plots\033[0m\n")

    for batch_size in BATCH_SIZES:
        # Load baseline data
        baseline = load_fragmentation_data(batch_size, "max", "no-op")
        if baseline is None:
            print(f"Warning: No baseline data for batch_size={batch_size}")
            continue

        for budget in BUDGETS:
            # Load strategy data
            strategy_data = {}
            for strategy in STRATEGIES:
                data = load_fragmentation_data(batch_size, budget, strategy)
                if data is not None:
                    strategy_data[strategy] = data

            if not strategy_data:
                print(f"Warning: No data for budget={budget}")
                continue

            filename = plot_fragmentation_over_time(
                batch_size, budget, baseline, strategy_data
            )
            print(f"Generated: {filename}")

    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
