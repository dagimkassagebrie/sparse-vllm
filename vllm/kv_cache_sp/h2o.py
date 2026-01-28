"""H2O (Heavy-Hitter Oracle) KV cache sparsification strategy.

This module implements the H2O algorithm for KV cache sparsification as described in:
"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf

H2O identifies and preserves "heavy-hitter" tokens (those with high cumulative
attention scores) along with recent tokens, evicting the rest to save memory.
"""

import math
from typing import Dict, List, Set

import numpy as np
import torch
from numpy.typing import NDArray

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.kv_cache_sp.base import (
    InternalStrategy,
    KVCacheSparsifierBase,
    KVCacheSparsifierStepOutput,
)
from vllm.outputs import RequestOutput


class H2OKVCacheSparsifier(KVCacheSparsifierBase):
    """H2O (Heavy-Hitter Oracle) KV cache sparsifier.

    H2O maintains two categories of tokens in the KV cache:
    1. Heavy-hitters: Tokens with high cumulative attention scores across
       all previous decoding steps (these are consistently important)
    2. Recent tokens: The most recently generated tokens (recency bias)

    When the KV cache exceeds the budget, H2O evicts tokens that are neither
    heavy-hitters nor recent, based on their cumulative attention scores.

    Attributes:
        budget: Maximum tokens to keep per sequence.
        num_per_evict: Tokens to evict per step when over budget.
        internal: Memory management strategy ("no-op", "free-block",
            "sparse-copy", or "spvllm").
        seq_ids_to_cum_attn_scores: Maps sequence IDs to their cumulative
            attention score arrays.

    Example:
        >>> sparsifier = H2OKVCacheSparsifier(
        ...     budget=512,
        ...     num_per_evict=1,
        ...     internal="spvllm"
        ... )
        >>> # During decoding, call step() with attention scores
        >>> output = sparsifier.step(block_manager, seq_id, attn_scores)

    Reference:
        Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
        Inference of Large Language Models", NeurIPS 2023.
    """

    def __init__(
        self,
        budget: int,
        num_per_evict: int,
        internal: InternalStrategy
    ) -> None:
        """Initialize the H2O sparsifier.

        Args:
            budget: Maximum number of tokens to retain in KV cache.
            num_per_evict: Number of tokens to evict when budget exceeded.
            internal: Memory management strategy for evicted slots.
        """
        super().__init__(budget, num_per_evict, internal)
        # Track cumulative attention scores per sequence
        self.seq_ids_to_cum_attn_scores: Dict[int, NDArray[np.floating]] = {}

    def _compute_tokens_to_keep(
        self,
        active_slots: NDArray[np.intp],
        cum_attn_scores: NDArray[np.floating]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Compute which tokens to keep based on H2O criteria.

        Splits the budget between heavy-hitters (high attention) and
        recent tokens (recency bias).

        Args:
            active_slots: Indices of currently active (non-evicted) slots.
            cum_attn_scores: Cumulative attention scores for all slots.

        Returns:
            Tuple of (topk_slots, recent_slots) - indices to preserve.
        """
        num_keep = self.budget - self.num_per_evict + 1
        k = num_keep // 2  # Heavy-hitters
        k_last = (num_keep + 1) // 2  # Recent tokens

        # Get top-k by attention score (excluding recent tokens from candidates)
        candidate_slots = active_slots[:-k_last]
        candidate_scores = cum_attn_scores[candidate_slots]
        topk_indices = np.argpartition(candidate_scores, -k)[-k:]
        topk_slots = candidate_slots[topk_indices]

        # Recent tokens are always kept
        recent_slots = active_slots[-k_last:]

        return topk_slots, recent_slots

    def _apply_eviction_strategy(
        self,
        block_manager: BlockSpaceManagerV1,
        seq_id: int,
        slots_to_evict: NDArray[np.intp],
        num_slots: int,
        block_size: int
    ) -> tuple[int, int, List[int], List[NDArray[np.bool_]]]:
        """Apply the configured eviction strategy.

        Args:
            block_manager: Block manager for KV cache operations.
            seq_id: Sequence identifier.
            slots_to_evict: Slot indices to evict.
            num_slots: Total number of slots.
            block_size: Size of each block.

        Returns:
            Tuple of (num_evicted_tokens, num_migrate_dst_blocks,
                     slots_to_migrate, updated_block_masks).
        """
        num_evicted_tokens = 0
        num_migrate_dst_blocks = 0
        slots_to_migrate: List[int] = []

        if self.internal == "no-op":
            # Simply mark slots as inactive, no memory reclamation
            block_manager.deactivate_slots(seq_id, slots_to_evict)

        elif self.internal == "free-block":
            # Deactivate slots and free fully empty blocks
            block_manager.deactivate_slots(seq_id, slots_to_evict)
            removed_blocks = block_manager.free_fully_deactivated_blocks(seq_id)
            # Update attention scores to match new block layout
            for i in removed_blocks:
                self.seq_ids_to_cum_attn_scores[seq_id] = np.delete(
                    self.seq_ids_to_cum_attn_scores[seq_id],
                    np.s_[i * block_size:(i + 1) * block_size]
                )
            num_evicted_tokens = len(removed_blocks) * block_size

        elif self.internal == "sparse-copy":
            # Copy preserved tokens to new compacted blocks
            self.seq_ids_to_cum_attn_scores[seq_id] = np.delete(
                self.seq_ids_to_cum_attn_scores[seq_id], slots_to_evict
            )
            num_evicted_tokens = len(slots_to_evict)
            num_migrate_dst_blocks = math.ceil(
                (num_slots - num_evicted_tokens + 1) / block_size
            )
            slots_to_migrate = np.setdiff1d(
                np.arange(num_slots), slots_to_evict
            ).tolist()

        elif self.internal == "spvllm":
            # SpvLLM: deactivate + free blocks + reuse slots for new tokens
            block_manager.deactivate_slots(seq_id, slots_to_evict)
            removed_blocks = block_manager.free_fully_deactivated_blocks(seq_id)
            for i in removed_blocks:
                self.seq_ids_to_cum_attn_scores[seq_id] = np.delete(
                    self.seq_ids_to_cum_attn_scores[seq_id],
                    np.s_[i * block_size:(i + 1) * block_size]
                )
            num_evicted_tokens = len(slots_to_evict)

        else:
            raise ValueError(
                f"Unrecognized KV cache internal memory management "
                f"strategy: {self.internal}"
            )

        block_masks = block_manager.block_tables[seq_id].masks()
        return num_evicted_tokens, num_migrate_dst_blocks, slots_to_migrate, block_masks

    def step(
        self,
        block_manager: BlockSpaceManagerV1,
        seq_id: int,
        attn_scores: torch.Tensor
    ) -> KVCacheSparsifierStepOutput:
        """Execute one H2O sparsification step.

        Accumulates attention scores, checks if budget is exceeded, and
        evicts low-importance tokens if necessary.

        Args:
            block_manager: Block manager for KV cache operations.
            seq_id: Unique sequence identifier.
            attn_scores: Attention scores of shape (num_layers, num_heads, num_tokens).

        Returns:
            KVCacheSparsifierStepOutput with eviction results and metrics.
        """
        num_slots = attn_scores.size(2)

        # Accumulate attention scores (mean across layers and heads)
        agg_attn_scores = attn_scores.numpy().mean(axis=(0, 1))
        if seq_id in self.seq_ids_to_cum_attn_scores:
            self.seq_ids_to_cum_attn_scores[seq_id].resize(num_slots)
            self.seq_ids_to_cum_attn_scores[seq_id] += agg_attn_scores
        else:
            self.seq_ids_to_cum_attn_scores[seq_id] = agg_attn_scores

        # Get current active slots from block masks
        block_masks = block_manager.block_tables[seq_id].masks()
        block_size = len(block_masks[0])
        total_block_mask = np.concatenate(block_masks)[:num_slots]
        active_slots = np.where(total_block_mask)[0]
        num_active_slots = len(active_slots)

        # Check if eviction is needed
        if num_active_slots <= self.budget:
            return KVCacheSparsifierStepOutput(
                do_evict=False,
                num_active_slots=num_active_slots,
                num_total_slots=len(block_masks) * block_size,
                num_evicted_tokens=0,
                num_migrate_dst_blocks=0,
                slots_to_migrate=[]
            )

        # Compute which tokens to keep (heavy-hitters + recent)
        topk_slots, recent_slots = self._compute_tokens_to_keep(
            active_slots, self.seq_ids_to_cum_attn_scores[seq_id]
        )

        # Compute slots to evict (everything else that's currently active)
        keep_mask = np.zeros(num_slots, dtype=np.bool_)
        keep_mask[topk_slots] = True
        keep_mask[recent_slots] = True
        slots_to_evict = np.where(~keep_mask & total_block_mask)[0]

        # Apply the configured eviction strategy
        (num_evicted_tokens, num_migrate_dst_blocks,
         slots_to_migrate, block_masks) = self._apply_eviction_strategy(
            block_manager, seq_id, slots_to_evict, num_slots, block_size
        )

        return KVCacheSparsifierStepOutput(
            do_evict=True,
            num_active_slots=num_active_slots,
            num_total_slots=len(block_masks) * block_size,
            num_evicted_tokens=num_evicted_tokens,
            num_migrate_dst_blocks=num_migrate_dst_blocks,
            slots_to_migrate=slots_to_migrate
        )

    def clean_self(self, outputs: List[RequestOutput]) -> None:
        """Clean up cumulative attention scores for completed sequences.

        Args:
            outputs: Completed request outputs containing sequence IDs to clean.
        """
        for output in outputs:
            for seq_id in output.seq_ids:
                self.seq_ids_to_cum_attn_scores.pop(seq_id, None)
