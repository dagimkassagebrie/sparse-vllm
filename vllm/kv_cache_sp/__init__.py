"""KV cache sparsification strategies for vLLM.

This module provides implementations of KV cache sparsification algorithms
that reduce memory usage by evicting less important tokens based on
attention scores.

Available Strategies:
    - H2OKVCacheSparsifier: Heavy-Hitter Oracle (recommended)
    - RandomKVCacheSparsifier: Random eviction (baseline only)

Example:
    >>> from vllm.kv_cache_sp import get_kv_cache_sparsifier
    >>> SparsifierClass = get_kv_cache_sparsifier("h2o")
    >>> sparsifier = SparsifierClass(budget=512, num_per_evict=1, internal="spvllm")
"""

from typing import Type

from vllm.kv_cache_sp.base import (
    InternalStrategy,
    KVCacheSparsifierBase,
    KVCacheSparsifierStepOutput,
)
from vllm.kv_cache_sp.h2o import H2OKVCacheSparsifier
from vllm.kv_cache_sp.random import RandomKVCacheSparsifier

# Registry of available sparsification methods
_SPARSIFIER_REGISTRY = {
    "random": RandomKVCacheSparsifier,
    "h2o": H2OKVCacheSparsifier,
}


def get_kv_cache_sparsifier(method: str) -> Type[KVCacheSparsifierBase]:
    """Get the sparsifier class for a given method name.

    Args:
        method: Name of the sparsification method ("h2o" or "random").

    Returns:
        The sparsifier class (not an instance).

    Raises:
        ValueError: If the method name is not recognized.
    """
    if method not in _SPARSIFIER_REGISTRY:
        valid_methods = ", ".join(f"'{m}'" for m in _SPARSIFIER_REGISTRY)
        raise ValueError(
            f"Unknown KV cache sparsification method: '{method}'. "
            f"Valid options are: {valid_methods}"
        )
    return _SPARSIFIER_REGISTRY[method]


def list_available_methods() -> list[str]:
    """Return list of available sparsification method names."""
    return list(_SPARSIFIER_REGISTRY.keys())


__all__ = [
    # Base classes
    "KVCacheSparsifierBase",
    "KVCacheSparsifierStepOutput",
    "InternalStrategy",
    # Implementations
    "H2OKVCacheSparsifier",
    "RandomKVCacheSparsifier",
    # Factory functions
    "get_kv_cache_sparsifier",
    "list_available_methods",
]
