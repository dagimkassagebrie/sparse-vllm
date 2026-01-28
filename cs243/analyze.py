#!/usr/bin/env python3
"""Analyze benchmark logs and generate performance metrics.

This script parses the custom CS243 log tags from vLLM server output and
computes statistical summaries including:
- System metrics (batch sizes, preemptions, GPU utilization)
- Attention operation timing
- Internal fragmentation ratios
- Sparse copying overhead

Usage:
    python analyze.py  # Analyzes all logs in cs243/logs/
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, TextIO, Optional

import numpy as np
from numpy.typing import NDArray

# Directory constants
CURRENT_DIR = Path(__file__).parent
LOGS_DIR = CURRENT_DIR / "logs"
RESULTS_DIR = CURRENT_DIR / "results"

# Analysis parameters
WARMUP_ITERATIONS = 200  # Skip initial iterations for fragmentation analysis
GPU_UTIL_WARMUP = 500  # Skip for GPU utilization
GPU_UTIL_COOLDOWN = 1000  # Skip final iterations

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class SystemMetrics:
    """Raw system metrics collected from logs."""
    num_scheduled_seqs: List[int] = field(default_factory=list)
    num_batched_tokens: List[int] = field(default_factory=list)
    num_blocks_to_swap_in: List[int] = field(default_factory=list)
    num_blocks_to_swap_out: List[int] = field(default_factory=list)
    num_blocks_to_copy: List[int] = field(default_factory=list)
    num_blocks_to_migrate: List[int] = field(default_factory=list)
    num_slots_to_migrate: List[int] = field(default_factory=list)
    running_queue_size: List[int] = field(default_factory=list)
    num_preempted: List[int] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)


@dataclass
class FragmentationMetrics:
    """Raw fragmentation metrics collected from logs."""
    num_active: List[int] = field(default_factory=list)
    num_total: List[int] = field(default_factory=list)


@dataclass
class AttentionMetrics:
    """Attention operation metrics."""
    op_count: int = 0
    total_time_us: int = 0

    @property
    def total_time_ms(self) -> float:
        """Total attention time in milliseconds."""
        return self.total_time_us / 1000.0

    @property
    def avg_time_us(self) -> float:
        """Average attention operation time in microseconds."""
        return self.total_time_us / self.op_count if self.op_count > 0 else 0.0


@dataclass
class CopyOverheadMetrics:
    """Sparse copy operation metrics."""
    op_count: int = 0
    total_time_us: int = 0

    @property
    def total_time_ms(self) -> float:
        """Total copy time in milliseconds."""
        return self.total_time_us / 1000.0


def compute_statistics(arr: NDArray) -> Dict[str, float]:
    """Compute comprehensive statistics for an array.

    Returns:
        Dictionary with mean, std, min, max, median, and percentiles.
    """
    if len(arr) == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
        }

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def parse_system_line(line: str, metrics: SystemMetrics) -> None:
    """Parse a #CS243S# system metrics log line."""
    parts = line[9:].split(",")
    metrics.num_scheduled_seqs.append(int(parts[0]))
    metrics.num_batched_tokens.append(int(parts[2]))
    metrics.num_blocks_to_swap_in.append(int(parts[3]))
    metrics.num_blocks_to_swap_out.append(int(parts[4]))
    metrics.num_blocks_to_copy.append(int(parts[5]))
    metrics.num_blocks_to_migrate.append(int(parts[6]))
    metrics.num_slots_to_migrate.append(int(parts[7]))
    metrics.running_queue_size.append(int(parts[11]))
    metrics.num_preempted.append(int(parts[12]))
    metrics.gpu_utilization.append(float(parts[13]))


def parse_fragmentation_line(line: str, metrics: FragmentationMetrics) -> None:
    """Parse a #CS243F# fragmentation log line."""
    num_active, num_total = line[9:].split(",")
    metrics.num_active.append(int(num_active))
    metrics.num_total.append(int(num_total))


def parse_log_file(f: TextIO) -> tuple[
    SystemMetrics, FragmentationMetrics, AttentionMetrics, CopyOverheadMetrics
]:
    """Parse all metrics from a log file.

    Args:
        f: Open file handle for the log file.

    Returns:
        Tuple of (system_metrics, frag_metrics, attn_metrics, copy_metrics).
    """
    sys_metrics = SystemMetrics()
    frag_metrics = FragmentationMetrics()
    attn_metrics = AttentionMetrics()
    copy_metrics = CopyOverheadMetrics()

    for line in f:
        if line.startswith("#CS243S#,"):
            parse_system_line(line, sys_metrics)
        elif line.startswith("#CS243A#,"):
            attn_metrics.op_count += 1
            attn_metrics.total_time_us += int(line[9:])
        elif line.startswith("#CS243F#,"):
            parse_fragmentation_line(line, frag_metrics)
        elif line.startswith("#CS243O#,"):
            copy_metrics.op_count += 1
            copy_metrics.total_time_us += int(line[9:])

    return sys_metrics, frag_metrics, attn_metrics, copy_metrics


def save_raw_metrics(name: str, sys_metrics: SystemMetrics,
                     frag_metrics: FragmentationMetrics) -> None:
    """Save raw metrics arrays to numpy files for later analysis."""
    # System metrics
    sys_path = RESULTS_DIR / f"{name}-sys.npy"
    with sys_path.open("wb") as fsys:
        for arr in [
            sys_metrics.num_scheduled_seqs,
            sys_metrics.num_batched_tokens,
            sys_metrics.num_blocks_to_swap_in,
            sys_metrics.num_blocks_to_swap_out,
            sys_metrics.num_blocks_to_copy,
            sys_metrics.num_blocks_to_migrate,
            sys_metrics.num_slots_to_migrate,
            sys_metrics.running_queue_size,
            sys_metrics.num_preempted,
            sys_metrics.gpu_utilization,
        ]:
            np.save(fsys, arr)

    # Fragmentation metrics (after warmup)
    frag_path = RESULTS_DIR / f"{name}-frag.npy"
    num_active = np.asarray(frag_metrics.num_active[WARMUP_ITERATIONS:])
    num_total = np.asarray(frag_metrics.num_total[WARMUP_ITERATIONS:])
    with frag_path.open("wb") as ffrag:
        np.save(ffrag, num_active)
        np.save(ffrag, num_total)


def analyze_metrics(
    sys_metrics: SystemMetrics,
    frag_metrics: FragmentationMetrics,
    attn_metrics: AttentionMetrics,
    copy_metrics: CopyOverheadMetrics
) -> Dict[str, Any]:
    """Compute summary statistics from raw metrics.

    Returns:
        Dictionary of computed statistics.
    """
    results = {}

    # Batched tokens statistics
    batched_stats = compute_statistics(np.array(sys_metrics.num_batched_tokens))
    results["num_batched_tokens_mean"] = batched_stats["mean"]
    results["num_batched_tokens_median"] = batched_stats["median"]
    results["num_batched_tokens_std"] = batched_stats["std"]
    results["num_batched_tokens_p99"] = batched_stats["p99"]

    # Preemption statistics
    results["num_preempted_total"] = sum(sys_metrics.num_preempted)
    results["num_preempted_rate"] = (
        sum(sys_metrics.num_preempted) / len(sys_metrics.num_preempted)
        if sys_metrics.num_preempted else 0.0
    )

    # GPU utilization (excluding warmup/cooldown)
    gpu_util = sys_metrics.gpu_utilization[GPU_UTIL_WARMUP:-GPU_UTIL_COOLDOWN]
    if gpu_util:
        gpu_stats = compute_statistics(np.array(gpu_util))
        results["gpu_utilization_mean"] = gpu_stats["mean"]
        results["gpu_utilization_std"] = gpu_stats["std"]
        results["gpu_utilization_p90"] = gpu_stats["p90"]
        results["gpu_utilization_p99"] = gpu_stats["p99"]
    else:
        results["gpu_utilization_mean"] = 0.0
        results["gpu_utilization_std"] = 0.0
        results["gpu_utilization_p90"] = 0.0
        results["gpu_utilization_p99"] = 0.0

    # Attention metrics
    results["attn_op_count"] = attn_metrics.op_count
    results["attn_total_time_ms"] = attn_metrics.total_time_ms
    results["attn_avg_time_us"] = attn_metrics.avg_time_us

    # Fragmentation analysis (after warmup)
    num_active = np.asarray(frag_metrics.num_active[WARMUP_ITERATIONS:])
    num_total = np.asarray(frag_metrics.num_total[WARMUP_ITERATIONS:])

    if len(num_total) > 0 and np.all(num_total > 0):
        frag_ratio = 1.0 - num_active / num_total
        frag_stats = compute_statistics(frag_ratio)
        results["frag_ratio_mean"] = frag_stats["mean"]
        results["frag_ratio_std"] = frag_stats["std"]
        results["frag_ratio_max"] = frag_stats["max"]
        results["frag_ratio_median"] = frag_stats["median"]
        results["frag_ratio_p99"] = frag_stats["p99"]
    else:
        results["frag_ratio_mean"] = 0.0
        results["frag_ratio_std"] = 0.0
        results["frag_ratio_max"] = 0.0
        results["frag_ratio_median"] = 0.0
        results["frag_ratio_p99"] = 0.0

    # Copy overhead metrics
    results["copy_op_count"] = copy_metrics.op_count
    results["copy_total_time_ms"] = copy_metrics.total_time_ms

    return results


def append_benchmark_metrics(metrics_path: Path, results: Dict[str, Any]) -> None:
    """Append metrics from benchmark_serving.py output.

    Also computes "effective" metrics normalized by GPU utilization.
    """
    if not metrics_path.exists():
        return

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    factor = results.get("gpu_utilization_mean", 1.0)
    if factor == 0:
        factor = 1.0

    # Raw metrics
    raw_keys = [
        "duration", "completed", "total_input_tokens", "total_output_tokens",
        "request_throughput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "std_itl_ms", "p99_itl_ms",
    ]
    for key in raw_keys:
        if key in metrics:
            results[f"raw_{key}"] = metrics[key]

    # Effective throughput (normalized by GPU util)
    throughput_keys = ["request_throughput", "output_throughput", "total_token_throughput"]
    for key in throughput_keys:
        if key in metrics:
            results[f"eff_{key}"] = metrics[key] / factor

    # Effective latency (normalized by GPU util)
    latency_keys = [
        "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "std_itl_ms", "p99_itl_ms",
    ]
    for key in latency_keys:
        if key in metrics:
            results[f"eff_{key}"] = metrics[key] * factor


def analyze_single_experiment(log_path: Path) -> Optional[Dict[str, Any]]:
    """Analyze a single experiment's logs.

    Args:
        log_path: Path to the stdout log file.

    Returns:
        Dictionary of results, or None if metrics file is missing.
    """
    name = log_path.name[7:-11]  # Remove "bench--" prefix and ".stdout.log" suffix
    metrics_path = LOGS_DIR / f"bench--{name}.metrics.json"

    if not metrics_path.exists():
        print(f"  Skipping {name}: metrics file not found")
        return None

    print(f"  Analyzing {name}...")

    # Parse log file
    with log_path.open("r", encoding="utf-8") as f:
        sys_metrics, frag_metrics, attn_metrics, copy_metrics = parse_log_file(f)

    # Save raw data
    save_raw_metrics(name, sys_metrics, frag_metrics)

    # Compute statistics
    results = analyze_metrics(sys_metrics, frag_metrics, attn_metrics, copy_metrics)

    # Add benchmark metrics
    append_benchmark_metrics(metrics_path, results)

    return results


def print_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary of analysis results."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Fragmentation: {results.get('frag_ratio_mean', 0):.2%} mean, "
              f"{results.get('frag_ratio_p99', 0):.2%} p99")
        print(f"  GPU Util: {results.get('gpu_utilization_mean', 0):.2%} mean")
        print(f"  Preemptions: {results.get('num_preempted_total', 0)}")
        if "raw_request_throughput" in results:
            print(f"  Throughput: {results['raw_request_throughput']:.2f} req/s")


def main() -> None:
    """Main entry point for log analysis."""
    print("\033[34;1m" + "=" * 60 + "\033[0m")
    print("\033[34;1mKV Cache Sparsification Benchmark Analysis\033[0m")
    print("\033[34;1m" + "=" * 60 + "\033[0m\n")

    all_results = {}

    log_files = list(LOGS_DIR.glob("bench--*.stdout.log"))
    if not log_files:
        print("No benchmark logs found in", LOGS_DIR)
        return

    print(f"Found {len(log_files)} benchmark logs to analyze\n")

    for log_path in sorted(log_files):
        results = analyze_single_experiment(log_path)
        if results is not None:
            name = log_path.name[7:-11]
            all_results[name] = results

    # Save all results
    output_path = RESULTS_DIR / "analyze.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
