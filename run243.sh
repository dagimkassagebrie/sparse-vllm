#!/bin/bash
#
# Run all CS243 KV cache sparsification experiments
#
# This script runs benchmarks for all combinations of:
#   - Batch sizes: 256, 2048
#   - Budgets: 256, 512, 1024, max (no eviction)
#   - Strategies: no-op, free-block, sparse-copy, spvllm
#
# Results are saved to cs243/logs/ and plots to cs243/plots/
#
# Usage: ./run243.sh

set -e  # Exit on error

echo "=========================================="
echo "SpvLLM Benchmark Suite"
echo "=========================================="

# Batch size 256 experiments
echo -e "\n[Batch Size 256]"

for budget in 256 512 1024; do
    for internal in no-op free-block sparse-copy spvllm; do
        echo "Running: budget=$budget, internal=$internal"
        python cs243/benchmark.py \
            --batch-size 256 \
            --sparse-kv-cache-num-per-evict 1 \
            --sparse-kv-cache-budget $budget \
            --sparse-kv-cache-internal $internal
    done
done

# Baseline (no sparsification)
echo "Running: baseline (no eviction)"
python cs243/benchmark.py \
    --batch-size 256 \
    --sparse-kv-cache-num-per-evict 1 \
    --sparse-kv-cache-budget max \
    --sparse-kv-cache-internal no-op

# Batch size 2048 experiments
echo -e "\n[Batch Size 2048]"

for budget in 256 512 1024; do
    for internal in no-op free-block sparse-copy spvllm; do
        echo "Running: budget=$budget, internal=$internal"
        python cs243/benchmark.py \
            --batch-size 2048 \
            --sparse-kv-cache-num-per-evict 1 \
            --sparse-kv-cache-budget $budget \
            --sparse-kv-cache-internal $internal
    done
done

# Baseline (no sparsification)
echo "Running: baseline (no eviction)"
python cs243/benchmark.py \
    --batch-size 2048 \
    --sparse-kv-cache-num-per-evict 1 \
    --sparse-kv-cache-budget max \
    --sparse-kv-cache-internal no-op

# Analysis and plotting
echo -e "\n[Analysis]"
echo "Analyzing results..."
python cs243/analyze.py

echo -e "\n[Plotting]"
echo "Generating plots..."
python cs243/plot_frag.py
python cs243/plot_metrics.py
python cs243/plot_motivation.py

echo -e "\n=========================================="
echo "Experiments complete!"
echo "Results: cs243/results/"
echo "Plots:   cs243/plots/"
echo "=========================================="
