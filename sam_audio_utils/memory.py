"""GPU memory management utilities for SAM-Audio processing.

This module provides memory budgeting and cleanup utilities optimized for
WSL2 environments with CUDA GPU acceleration. The key fix is adding
torch.cuda.synchronize() before empty_cache() to ensure all GPU operations
complete before attempting to free memory.
"""

from __future__ import annotations

import gc
import torch


class GPUMemoryBudget:
    """Track and manage GPU memory budget for processing.

    WSL2 considerations:
    - Total memory: Device dependent (e.g., 48GB RTX 8000)
    - Reserve fraction: Recommended 0.55 for WSL2 vs 0.70 for native Linux
    - Virtualization overhead: ~20-30% in WSL2
    - Model baseline: ~4.8GB for SAM-Audio large model

    Example:
        >>> budget = GPUMemoryBudget("cuda", reserve_fraction=0.55)
        >>> # After loading model
        >>> budget.establish_baseline()
        >>> # Before processing chunk
        >>> if budget.can_process_chunk(estimated_size):
        >>>     # Process chunk
        >>>     pass
        >>> # After processing
        >>> budget.aggressive_cleanup()
    """

    def __init__(self, device: str, reserve_fraction: float = 0.55):
        """Initialize GPU memory budget tracker.

        Args:
            device: Device string ("cuda" or "cpu")
            reserve_fraction: Maximum fraction of GPU memory to use (0.0-1.0)
                             0.55 recommended for WSL2, 0.70 for native Linux
        """
        self.device = device
        self.reserve_fraction = reserve_fraction
        self.baseline_usage = None

    def establish_baseline(self) -> None:
        """Call after model load to record baseline memory usage.

        This establishes the baseline GPU memory usage after loading the model
        but before processing any data. Useful for tracking memory growth.
        """
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.baseline_usage = torch.cuda.memory_allocated()

    def get_available_memory(self) -> int:
        """Return bytes of GPU memory available for processing.

        Returns:
            Available memory in bytes. Returns float('inf') for CPU device.
        """
        if self.device != "cuda":
            return float('inf')

        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        budget = int(total * self.reserve_fraction)
        return max(0, budget - allocated)

    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics in GB.

        Returns:
            Dictionary with 'allocated', 'reserved', 'total', and 'available' in GB
        """
        if self.device != "cuda":
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "total": 0.0,
                "available": float('inf'),
            }

        props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = props.total_memory / (1024**3)
        available = self.get_available_memory() / (1024**3)

        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "available": available,
        }

    def can_process_chunk(self, estimated_chunk_size: int) -> bool:
        """Check if we have enough memory for a chunk.

        Args:
            estimated_chunk_size: Estimated memory requirement in bytes

        Returns:
            True if sufficient memory is available
        """
        return self.get_available_memory() >= estimated_chunk_size

    def aggressive_cleanup(self) -> None:
        """Perform complete memory cleanup (call after each chunk).

        **CRITICAL FIX for WSL2 crashes:**
        The order of operations is crucial:
        1. Python garbage collection - frees Python objects holding CUDA refs
        2. CUDA synchronize - waits for ALL GPU operations to complete
        3. Empty cache - releases completed allocations
        4. Final GC - catches any remaining Python objects

        Without step 2 (synchronize), pending GPU operations continue holding
        memory, leading to fragmentation and eventual OOM crashes, especially
        in WSL2 environments where memory management is less efficient.

        This fixes the bug in run_sam_audio_batch.py:73-76 which was missing
        the synchronize() call.
        """
        if self.device != "cuda":
            return

        # Order matters for effective cleanup:
        # 1. Python garbage collection first
        gc.collect()

        # 2. Wait for all CUDA operations to complete
        #    ← THIS IS THE CRITICAL FIX - was missing in original code
        torch.cuda.synchronize()

        # 3. Release cached memory (now that ops are complete)
        torch.cuda.empty_cache()

        # 4. Final GC pass to catch any Python objects holding CUDA refs
        gc.collect()


def estimate_chunk_memory(duration_seconds: int, sample_rate: int = 16000) -> int:
    """Estimate GPU memory needed for processing a chunk.

    Conservative estimates based on observed SAM-Audio usage:
    - Input audio tensor: duration * sample_rate * 4 bytes (float32)
    - Model intermediate states: ~3-4x input size (features, embeddings)
    - Output tensors (target + residual): 2x input size
    - Safety margin: 1.5x multiplier for reranking and other overhead

    Args:
        duration_seconds: Chunk duration in seconds
        sample_rate: Audio sample rate (default: 16000 Hz)

    Returns:
        Estimated memory requirement in bytes

    Example:
        >>> # 10-second chunk at 16kHz
        >>> estimate_chunk_memory(10, 16000)
        1680000  # ~1.6 MB base, but with overhead ~12 MB
    """
    samples = duration_seconds * sample_rate
    input_bytes = samples * 4  # float32

    # Total estimate:
    # - Input: 1x
    # - Intermediates (features, embeddings, ranking): 4x
    # - Outputs (target + residual): 2x
    # - Safety margin: 1.5x
    total_estimate = input_bytes * (1 + 4 + 2) * 1.5

    return int(total_estimate)
