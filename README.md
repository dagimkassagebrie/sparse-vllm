# SpvLLM: Efficient KV Cache Sparsification in vLLM

> Harvard CS243 (Fall 2024) Final Project

**Authors:** Yao Xiao, Dagim Gebrie

## Overview

SpvLLM implements efficient KV cache sparsification in vLLM, reducing memory fragmentation caused by token eviction during LLM inference. Our approach achieves:

- **Up to 55.7% reduction** in internal fragmentation
- **Up to 2.21× higher throughput** compared to baseline vLLM
- **Up to 48.9% lower latency** for inter-token generation

## The Problem

When using KV cache sparsification to evict less important tokens, internal fragmentation occurs because evicted slots leave "holes" in the KV cache blocks. Existing approaches have significant limitations:

| Strategy | Description | Limitation |
|----------|-------------|------------|
| **no-op** | Mark slots inactive, don't reclaim | High fragmentation |
| **free-block** | Free blocks only when fully empty | Moderate fragmentation |
| **sparse-copy** | Copy active tokens to new blocks | Copy overhead, causes preemption |

## Our Solution: SpvLLM

SpvLLM leverages a key insight: **attention computation is order-agnostic**. The scaled dot-product attention is:

```
output = Σ softmax(QK^T/√d) × V
```

Since this is a weighted sum, the order of tokens in the KV cache doesn't affect the result. Therefore, **new tokens can directly fill evicted slots** instead of requiring sequential allocation or copying.

### How It Works

1. When a token is evicted, mark its slot as "deactivated" (available for reuse)
2. When a new token arrives, check for deactivated slots
3. If found, write the new token's KV cache to that slot instead of allocating new space
4. Update the slot mapping to point to the reused slot

This eliminates fragmentation without copying overhead.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Scheduler                             │
│  - Collects reused_slot info from sequences                 │
│  - Passes to SequenceGroupMetadata                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Block Manager                            │
│  - append_slots() detects available deactivated slots       │
│  - Computes physical slot address for reuse                 │
│  - Sets seq.reused_slot with the address                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Attention Backend                         │
│  - Uses reused_slot for slot_mapping instead of sequential  │
│  - KV cache written to correct physical location            │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
sparse-vllm/
├── vllm/
│   ├── kv_cache_sp/           # KV cache sparsification strategies
│   │   ├── base.py            # Base class and output dataclass
│   │   ├── h2o.py             # H2O (Heavy-Hitter Oracle) implementation
│   │   └── random.py          # Random eviction baseline
│   ├── core/
│   │   ├── block_manager_v1.py # Block management with slot reuse
│   │   └── scheduler.py       # Integration with vLLM scheduler
│   ├── attention/
│   │   └── backends/utils.py  # Slot mapping with reuse support
│   └── config.py              # Sparsification configuration
├── csrc/
│   └── attention/
│       └── attention_kernels.cu # CUDA kernels with block masks
├── cs243/
│   ├── benchmark.py           # Experiment orchestration
│   ├── analyze.py             # Log parsing and metrics
│   ├── plot_metrics.py        # Visualization generation
│   └── plot_frag.py           # Fragmentation plots
└── ResearchReport.pdf         # Full research paper
```

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Experiments](#experiments)
- [AWS Setup](#aws-setup)

## Installation

### Prerequisites

- CUDA 12.x
- Python 3.9+
- NVIDIA GPU with compute capability 7.0+

### Install from Source

```bash
# Clone the repository
git clone https://github.com/dagimkassagebrie/sparse-vllm.git
cd sparse-vllm

# Install vLLM with SpvLLM modifications
pip install -v -e .

# Install experiment dependencies
pip install -r requirements-dev.txt
pip install seaborn
```

## Quick Start

### Using SpvLLM with vLLM Server

```bash
vllm serve facebook/opt-125m \
    --sparse-kv-cache-method h2o \
    --sparse-kv-cache-budget 512 \
    --sparse-kv-cache-internal spvllm
```

### Programmatic Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    sparse_kv_cache_method="h2o",
    sparse_kv_cache_budget=512,
    sparse_kv_cache_internal="spvllm",
)

prompts = ["Hello, my name is"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--sparse-kv-cache-method` | Sparsification algorithm (`h2o`, `random`, or `None`) | `None` |
| `--sparse-kv-cache-budget` | Max tokens to keep in KV cache | `2048` |
| `--sparse-kv-cache-num-per-evict` | Tokens to evict per step | `1` |
| `--sparse-kv-cache-internal` | Memory strategy (`no-op`, `free-block`, `sparse-copy`, `spvllm`) | `spvllm` |

### Internal Strategy Comparison

| Strategy | Fragmentation | Overhead | Best For |
|----------|--------------|----------|----------|
| `no-op` | High | None | Testing only |
| `free-block` | Medium | Low | Memory-constrained |
| `sparse-copy` | Low | High | Not recommended |
| `spvllm` | **Low** | **Minimal** | **Production use** |

## Experiments

### Download Dataset

```bash
wget -O ./benchmarks/sharegpt.json \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Run Benchmarks

```bash
# Run all experiments
./run243.sh

# Or run individual benchmark
python cs243/benchmark.py \
    --sparse-kv-cache-method h2o \
    --sparse-kv-cache-budget 512 \
    --sparse-kv-cache-internal spvllm

# Analyze results
python cs243/analyze.py

# Generate plots
python cs243/plot_metrics.py
```

Results are saved to `cs243/logs/` and plots to `cs243/plots/`.

## AWS Setup

<details>
<summary>Click to expand AWS setup instructions</summary>

### Request GPU Quota

Go to [EC2 Service Quotas](https://us-east-2.console.aws.amazon.com/servicequotas/home/services/ec2/quotas) and request increase for "Running On-Demand G and VT instances" to at least 8 vCPUs.

### Launch Instance

1. Go to [EC2 Dashboard](https://us-east-2.console.aws.amazon.com/ec2/home)
2. Launch instance with:
   - AMI: Ubuntu Server 22.04 LTS
   - Instance type: `g4dn.2xlarge` or larger
   - Storage: 60 GiB

### Install Dependencies

```bash
# System packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y ccache

# Conda
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc

# CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-6

# NVIDIA Driver
sudo apt-get install -y nvidia-open
sudo reboot
```

### Environment Variables

Add to `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export MAX_JOBS=4
export VLLM_ATTENTION_BACKEND=XFORMERS
export DISABLE_MOE_BUILD=1
```

</details>

## Citation

If you use SpvLLM in your research, please cite:

```bibtex
@misc{spvllm2024,
  title={SpvLLM: Efficient KV Cache Sparsification in vLLM},
  author={Xiao, Yao and Gebrie, Dagim},
  year={2024},
  institution={Harvard University},
  note={CS243 Fall 2024 Final Project}
}
```

## License

This project builds on [vLLM](https://github.com/vllm-project/vllm) and is subject to its Apache 2.0 license.

## Acknowledgments

We thank Professor Minlan Yu for her guidance and mentorship throughout this project.
