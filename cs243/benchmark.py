#!/usr/bin/env python3
"""Run benchmark experiments for KV cache sparsification strategies.

This script orchestrates benchmark experiments by:
1. Starting a vLLM server with specified KV cache sparsification settings
2. Running the benchmark client against the server
3. Collecting and saving performance metrics

Usage:
    python benchmark.py --sparse-kv-cache-internal spvllm --sparse-kv-cache-budget 512
    python benchmark.py --dataset random --random-input-len 512 --random-output-len 256
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

# Directory constants
CURRENT_DIR = Path(__file__).parent
ROOT_DIR = CURRENT_DIR.parent
BENCH_DIR = ROOT_DIR / "benchmarks"
LOGS_DIR = CURRENT_DIR / "logs"

# Server configuration
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
SERVER_STARTUP_WAIT = 10  # seconds
HEALTH_CHECK_RETRIES = 12
HEALTH_CHECK_INTERVAL = 5  # seconds
SERVER_SHUTDOWN_WAIT = 10  # seconds

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model: str
    batch_size: int
    dataset: str
    sharegpt_path: Optional[str]
    random_input_len: int
    random_output_len: int
    random_range_ratio: float
    random_prefix_len: int
    sparse_method: str
    sparse_budget: int
    sparse_num_per_evict: int
    sparse_internal: str
    force: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BenchmarkConfig":
        """Create config from parsed arguments."""
        return cls(
            model=args.model,
            batch_size=args.batch_size,
            dataset=args.dataset,
            sharegpt_path=args.sharegpt_path,
            random_input_len=args.random_input_len,
            random_output_len=args.random_output_len,
            random_range_ratio=args.random_range_ratio,
            random_prefix_len=args.random_prefix_len,
            sparse_method=args.sparse_kv_cache_method,
            sparse_budget=args.sparse_kv_cache_budget,
            sparse_num_per_evict=args.sparse_kv_cache_num_per_evict,
            sparse_internal=args.sparse_kv_cache_internal,
            force=args.force,
        )

    def get_experiment_id(self) -> str:
        """Generate a unique identifier for this experiment configuration."""
        parts = [self.batch_size, self.dataset]
        if self.dataset == "sharegpt":
            parts.append(self.sharegpt_path)
        elif self.dataset == "random":
            parts.extend([
                self.random_input_len,
                self.random_output_len,
                self.random_range_ratio,
                self.random_prefix_len,
            ])
        parts.extend([
            self.sparse_method,
            self.sparse_budget,
            self.sparse_num_per_evict,
            self.sparse_internal,
        ])
        return "bench--" + "-".join(str(p) for p in parts)


def print_colored(message: str, color: str = "default") -> None:
    """Print a message with ANSI color codes."""
    colors = {
        "default": "\033[0m",
        "red": "\033[31;1m",
        "green": "\033[32;1m",
        "yellow": "\033[33;1m",
        "blue": "\033[34;1m",
        "gray": "\033[90m",
        "bold": "\033[1m",
    }
    reset = "\033[0m"
    print(f"{colors.get(color, '')}{message}{reset}")


def validate_dataset(config: BenchmarkConfig) -> Optional[Path]:
    """Validate that the required dataset exists.

    Returns:
        Path to sharegpt dataset if using sharegpt, None otherwise.

    Raises:
        FileNotFoundError: If sharegpt dataset doesn't exist.
    """
    if config.dataset == "sharegpt":
        sharegpt_path = BENCH_DIR / config.sharegpt_path
        if not sharegpt_path.exists():
            raise FileNotFoundError(
                f"ShareGPT dataset not found at: {sharegpt_path}"
            )
        return sharegpt_path
    return None


def check_existing_results(config: BenchmarkConfig) -> tuple[Path, Path, Path]:
    """Check for existing benchmark results.

    Returns:
        Tuple of (stdout_path, stderr_path, metrics_path).

    Raises:
        FileExistsError: If results exist and force is False.
    """
    experiment_id = config.get_experiment_id()
    stdout_path = LOGS_DIR / f"{experiment_id}.stdout.log"
    stderr_path = LOGS_DIR / f"{experiment_id}.stderr.log"
    metrics_path = LOGS_DIR / f"{experiment_id}.metrics.json"

    if (not config.force and stdout_path.exists()
            and stderr_path.exists() and metrics_path.exists()):
        raise FileExistsError(
            f"Benchmark outputs already exist:\n"
            f"  Stdout:  {stdout_path}\n"
            f"  Stderr:  {stderr_path}\n"
            f"  Metrics: {metrics_path}\n"
            f"Use --force to overwrite."
        )

    return stdout_path, stderr_path, metrics_path


def build_server_options(config: BenchmarkConfig) -> List[str]:
    """Build command-line options for the vLLM server."""
    return [
        "--enforce-eager",
        "--gpu-memory-utilization", str(DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(config.batch_size),
        "--sparse-kv-cache-method", config.sparse_method,
        "--sparse-kv-cache-budget", str(config.sparse_budget),
        "--sparse-kv-cache-num-per-evict", str(config.sparse_num_per_evict),
        "--sparse-kv-cache-internal", config.sparse_internal,
    ]


def build_client_options(
    config: BenchmarkConfig,
    metrics_path: Path,
    sharegpt_path: Optional[Path]
) -> List[str]:
    """Build command-line options for the benchmark client."""
    return [
        "--save-result",
        "--result-filename", str(metrics_path),
        "--backend", "vllm",
        "--model", config.model,
        "--dataset-name", config.dataset,
        "--dataset-path", str(sharegpt_path) if sharegpt_path else "",
        "--random-input-len", str(config.random_input_len),
        "--random-output-len", str(config.random_output_len),
        "--random-range-ratio", str(config.random_range_ratio),
        "--random-prefix-len", str(config.random_prefix_len),
        "--request-rate", "inf",
        "--num-prompts", "1000",
    ]


def start_server(
    config: BenchmarkConfig,
    stdout_file,
    stderr_file
) -> subprocess.Popen:
    """Start the vLLM server process."""
    server_options = build_server_options(config)
    env = {
        **os.environ,
        "VLLM_LOGGING_LEVEL": "ERROR",
        "VLLM_CS243_PRINT_BENCHMARK": "1",
    }

    print_colored("Starting up server...", "gray")
    return subprocess.Popen(
        ["vllm", "serve", config.model, *server_options],
        cwd=ROOT_DIR,
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
    )


def wait_for_server_health() -> bool:
    """Wait for the server to become healthy.

    Returns:
        True if server is healthy, False if max retries exceeded.
    """
    time.sleep(SERVER_STARTUP_WAIT)

    for attempt in range(1, HEALTH_CHECK_RETRIES + 1):
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print_colored(f"[Att#{attempt}] Server up", "bold")
                return True
            print_colored(
                f"[Att#{attempt}] Server not ready ({response.status_code})",
                "gray"
            )
        except requests.exceptions.ConnectionError:
            print_colored(
                f"[Att#{attempt}] Server not ready (connection error)",
                "gray"
            )

        time.sleep(HEALTH_CHECK_INTERVAL)

    return False


def run_benchmark_client(
    config: BenchmarkConfig,
    metrics_path: Path,
    sharegpt_path: Optional[Path]
) -> int:
    """Run the benchmark client against the server.

    Returns:
        Client process return code.
    """
    client_options = build_client_options(config, metrics_path, sharegpt_path)

    result = subprocess.run(
        ["python", "benchmarks/benchmark_serving.py", *client_options],
        cwd=ROOT_DIR,
        env=os.environ,
    )
    return result.returncode


def shutdown_server(server_proc: subprocess.Popen) -> None:
    """Gracefully shut down the server process."""
    print_colored("Shutting down server...", "gray")
    server_proc.terminate()
    time.sleep(SERVER_SHUTDOWN_WAIT)


def run_benchmark(config: BenchmarkConfig) -> None:
    """Execute the full benchmark workflow."""
    print_colored("**********", "blue")
    print(config)

    # Validate dataset
    try:
        sharegpt_path = validate_dataset(config)
    except FileNotFoundError as e:
        print_colored(f"ERROR: {e}", "red")
        return

    # Check for existing results
    try:
        stdout_path, stderr_path, metrics_path = check_existing_results(config)
    except FileExistsError as e:
        print_colored(f"SKIPPED: {e}", "yellow")
        return

    # Open log files
    with (stdout_path.open("w", encoding="utf-8") as fout,
          stderr_path.open("w", encoding="utf-8") as ferr):

        # Start server
        server_proc = start_server(config, fout, ferr)

        try:
            # Wait for server to be ready
            if not wait_for_server_health():
                print_colored("ERROR: Server failed to start", "red")
                return

            # Run benchmark
            returncode = run_benchmark_client(config, metrics_path, sharegpt_path)
            if returncode != 0:
                print_colored(f"ERROR: Benchmark client failed with code {returncode}", "red")
                return

        finally:
            # Always shut down the server
            shutdown_server(server_proc)

    # Print output locations
    print(f"Stdout:  ", end="")
    print_colored(str(stdout_path), "gray")
    print(f"Stderr:  ", end="")
    print_colored(str(stderr_path), "gray")
    print(f"Metrics: ", end="")
    print_colored(str(metrics_path), "gray")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run KV cache sparsification benchmark experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing benchmark results",
    )

    # Experiment setup arguments
    exp_group = parser.add_argument_group("experiment setup")
    exp_group.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to benchmark",
    )
    exp_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences",
    )
    exp_group.add_argument(
        "--dataset",
        type=str,
        choices=["sharegpt", "random"],
        default="sharegpt",
        help="Dataset type for benchmarking",
    )

    # ShareGPT dataset options
    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-path",
        type=str,
        default="sharegpt.json",
        help="Dataset path relative to /benchmarks",
    )

    # Random dataset options
    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument("--random-input-len", type=int, default=1024)
    random_group.add_argument("--random-output-len", type=int, default=128)
    random_group.add_argument("--random-range-ratio", type=float, default=1.0)
    random_group.add_argument("--random-prefix-len", type=int, default=0)

    # KV cache sparsification arguments
    sparse_group = parser.add_argument_group("KV cache sparsification")
    sparse_group.add_argument(
        "--sparse-kv-cache-method",
        type=str,
        choices=["random", "h2o"],
        default="h2o",
        help="Sparsification algorithm",
    )
    sparse_group.add_argument(
        "--sparse-kv-cache-budget",
        type=lambda val: int(val) if val != "max" else sys.maxsize,
        default=512,
        help="Maximum tokens to keep in KV cache (use 'max' for no limit)",
    )
    sparse_group.add_argument(
        "--sparse-kv-cache-num-per-evict",
        type=int,
        default=1,
        help="Tokens to evict per sparsification step",
    )
    sparse_group.add_argument(
        "--sparse-kv-cache-internal",
        type=str,
        choices=["no-op", "free-block", "sparse-copy", "spvllm"],
        default="spvllm",
        help="Internal memory management strategy",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Validate max budget requires no-op
    if args.sparse_kv_cache_budget == sys.maxsize:
        if args.sparse_kv_cache_internal != "no-op":
            print_colored(
                "ERROR: --sparse-kv-cache-budget=max requires "
                "--sparse-kv-cache-internal=no-op",
                "red"
            )
            return

    config = BenchmarkConfig.from_args(args)
    run_benchmark(config)


if __name__ == "__main__":
    main()
