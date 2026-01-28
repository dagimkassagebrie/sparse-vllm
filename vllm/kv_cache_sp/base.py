"""Base classes for KV cache sparsification strategies.

This module provides the abstract base class and data structures for implementing
KV cache sparsification strategies in vLLM. Sparsification reduces memory usage
by evicting less important tokens from the KV cache based on attention scores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal

import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.outputs import RequestOutput

# Valid internal memory management strategies for KV cache sparsification
InternalStrategy = Literal["no-op", "free-block", "sparse-copy", "spvllm"]


@dataclass
class KVCacheSparsifierStepOutput:
    """Output from a single sparsification step.

    This dataclass captures the results of one iteration of KV cache
    sparsification, including eviction status and fragmentation metrics.

    Attributes:
        do_evict: Whether an eviction was performed in this step.
        num_active_slots: Number of slots currently holding active (non-evicted)
            tokens in the KV cache.
        num_total_slots: Total number of slots allocated in the KV cache blocks.
            The difference (num_total_slots - num_active_slots) represents
            internal fragmentation.
        num_evicted_tokens: Number of tokens physically removed from the block
            table layout. Note: This differs from masked slots which remain
            allocated but inactive.
        num_migrate_dst_blocks: Number of destination blocks needed for migration
            when using the sparse-copy strategy.
        slots_to_migrate: List of slot indices that should be copied during
            migration (used by sparse-copy strategy).
    """
    do_evict: bool
    num_active_slots: int
    num_total_slots: int
    num_evicted_tokens: int
    num_migrate_dst_blocks: int
    slots_to_migrate: List[int]

    @property
    def fragmentation_ratio(self) -> float:
        """Calculate the internal fragmentation ratio.

        Returns:
            Ratio of inactive slots to total slots (0.0 to 1.0).
            Higher values indicate more wasted memory.
        """
        if self.num_total_slots == 0:
            return 0.0
        return 1.0 - (self.num_active_slots / self.num_total_slots)


class KVCacheSparsifierBase(ABC):
    """Abstract base class for KV cache sparsifiers.

    KV cache sparsifiers reduce memory usage during LLM inference by selectively
    evicting less important tokens from the KV cache. Different strategies
    (H2O, SnapKV, etc.) use different criteria to determine which tokens to evict.

    Attributes:
        budget: Maximum number of tokens to keep in the KV cache per sequence.
        num_per_evict: Number of tokens to evict per sparsification step.
        internal: Memory management strategy for handling evicted slots.
            One of: "no-op", "free-block", "sparse-copy", "spvllm".

    Example:
        >>> sparsifier = H2OKVCacheSparsifier(budget=512, num_per_evict=1,
        ...                                    internal="spvllm")
        >>> output = sparsifier.step(block_manager, seq_id, attn_scores)
        >>> if output.do_evict:
        ...     print(f"Evicted tokens, fragmentation: {output.fragmentation_ratio:.2%}")
    """

    def __init__(
        self,
        budget: int,
        num_per_evict: int,
        internal: InternalStrategy
    ) -> None:
        """Initialize the KV cache sparsifier.

        Args:
            budget: Maximum number of tokens to keep in KV cache per sequence.
                Must be positive.
            num_per_evict: Number of tokens to evict when budget is exceeded.
                Must be positive and <= budget.
            internal: Memory management strategy. Options:
                - "no-op": Mark slots inactive but don't reclaim memory
                - "free-block": Free fully deactivated blocks
                - "sparse-copy": Copy active tokens to new compacted blocks
                - "spvllm": Reuse deactivated slots for new tokens

        Raises:
            ValueError: If budget or num_per_evict are invalid.
        """
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        if num_per_evict <= 0:
            raise ValueError(f"num_per_evict must be positive, got {num_per_evict}")
        if num_per_evict > budget:
            raise ValueError(
                f"num_per_evict ({num_per_evict}) cannot exceed budget ({budget})"
            )

        self.budget = budget
        self.num_per_evict = num_per_evict
        self.internal = internal

    @abstractmethod
    def step(
        self,
        block_manager: BlockSpaceManagerV1,
        seq_id: int,
        attn_scores: torch.Tensor
    ) -> KVCacheSparsifierStepOutput:
        """Execute one sparsification step.

        Analyzes attention scores and evicts low-importance tokens if the
        KV cache budget is exceeded. The specific eviction criteria depend
        on the sparsifier implementation.

        Args:
            block_manager: Block space manager handling KV cache blocks.
                Currently only v1 block manager is supported.
            seq_id: Unique identifier for the sequence being processed.
            attn_scores: Attention scores tensor of shape
                (num_layers, num_heads, num_tokens). Used to determine
                token importance for eviction decisions.

        Returns:
            KVCacheSparsifierStepOutput containing eviction status and metrics.

        Note:
            Subclasses must implement this method with their specific
            eviction strategy (e.g., H2O keeps heavy-hitters and recent tokens).
        """
        raise NotImplementedError

    @abstractmethod
    def clean_self(self, outputs: List[RequestOutput]) -> None:
        """Clean up internal state for completed requests.

        Sparsifiers may maintain per-sequence state (e.g., cumulative attention
        scores). This method removes state for sequences that have completed
        to prevent memory leaks.

        Args:
            outputs: List of completed request outputs containing sequence IDs
                that can be cleaned up.

        Note:
            Subclasses must implement this method, even if it's a no-op.
            Failure to clean up can cause memory leaks for long-running servers.
        """
        raise NotImplementedError
