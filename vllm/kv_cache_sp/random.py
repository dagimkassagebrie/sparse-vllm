"""Random KV cache sparsification strategy (baseline for comparison)."""

import math
from typing import List

import numpy as np
import torch

from vllm.core.block_manager_v1 import BlockSpaceManagerV1
from vllm.kv_cache_sp.base import (
    KVCacheSparsifierBase,
    KVCacheSparsifierStepOutput,
)
from vllm.outputs import RequestOutput


class RandomKVCacheSparsifier(KVCacheSparsifierBase):
    """Random KV cache sparsifier for baseline comparison.

    Randomly evicts tokens when exceeding KV cache budget. This is intended
    only for experimental comparison - random eviction performs poorly in
    practice since it ignores token importance.
    """

    def step(
        self,
        block_manager: BlockSpaceManagerV1,
        seq_id: int,
        attn_scores: torch.Tensor
    ) -> KVCacheSparsifierStepOutput:
        """Execute one random sparsification step."""
        num_slots = attn_scores.size(2)

        # Get current active slots from block masks
        block_masks = block_manager.block_tables[seq_id].masks()
        block_size = len(block_masks[0])
        total_block_mask = np.concatenate(block_masks)[:num_slots]
        active_slots = np.where(total_block_mask)[0]
        num_active_slots = len(active_slots)

        if num_active_slots <= self.budget:
            return KVCacheSparsifierStepOutput(
                do_evict=False,
                num_active_slots=num_active_slots,
                num_total_slots=len(block_masks) * block_size,
                num_evicted_tokens=0,
                num_migrate_dst_blocks=0,
                slots_to_migrate=[],
            )

        # Randomly select slots to evict
        slots_to_evict = np.random.choice(
            active_slots, self.num_per_evict, replace=False
        )

        num_evicted_tokens = 0
        num_migrate_dst_blocks = 0
        slots_to_migrate: List[int] = []

        if self.internal == "no-op":
            block_manager.deactivate_slots(seq_id, slots_to_evict)

        elif self.internal == "free-block":
            block_manager.deactivate_slots(seq_id, slots_to_evict)
            removed_blocks = block_manager.free_fully_deactivated_blocks(seq_id)
            block_masks = block_manager.block_tables[seq_id].masks()
            num_evicted_tokens = len(removed_blocks) * block_size

        elif self.internal == "sparse-copy":
            num_evicted_tokens = len(slots_to_evict)
            num_migrate_dst_blocks = math.ceil(
                (num_slots - num_evicted_tokens + 1) / block_size
            )
            slots_to_migrate = np.setdiff1d(
                np.arange(num_slots), slots_to_evict
            ).tolist()

        elif self.internal == "spvllm":
            block_manager.deactivate_slots(seq_id, slots_to_evict)
            removed_blocks = block_manager.free_fully_deactivated_blocks(seq_id)
            block_masks = block_manager.block_tables[seq_id].masks()
            num_evicted_tokens = len(slots_to_evict)

        else:
            raise ValueError(
                f"Unrecognized KV cache internal strategy: {self.internal}"
            )

        return KVCacheSparsifierStepOutput(
            do_evict=True,
            num_active_slots=num_active_slots,
            num_total_slots=len(block_masks) * block_size,
            num_evicted_tokens=num_evicted_tokens,
            num_migrate_dst_blocks=num_migrate_dst_blocks,
            slots_to_migrate=slots_to_migrate,
        )

    def clean_self(self, outputs: List[RequestOutput]) -> None:
        """No cleanup needed for random sparsifier."""
        pass
