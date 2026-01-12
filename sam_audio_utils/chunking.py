"""Adaptive chunking logic for SAM-Audio processing.

This module provides intelligent chunk size calculation based on file duration
and available GPU memory to optimize processing efficiency while preventing
out-of-memory crashes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import GPUMemoryBudget


class AdaptiveChunker:
    """Determine optimal chunk size based on file duration and memory.

    The chunking strategy balances processing overhead against memory safety:
    - Very short files (<30s): No chunking (single-pass processing)
    - Medium files (30-300s): Default chunk size (typically 10s)
    - Long files (>300s): Larger chunks to reduce concatenation overhead

    This prevents excessive chunking for long files while maintaining safety
    for files within the typical use case.

    Example:
        >>> chunker = AdaptiveChunker(default_chunk_size=10)
        >>> # 2-minute file
        >>> size = chunker.calculate_chunk_size(120, memory_budget)
        >>> size
        10  # Uses default
        >>> # 10-minute file
        >>> size = chunker.calculate_chunk_size(600, memory_budget)
        >>> size
        20  # Adaptive: larger chunks to reduce overhead
    """

    def __init__(
        self,
        default_chunk_size: int = 10,
        min_chunk_size: int = 5,
        max_chunk_size: int = 60,
    ):
        """Initialize adaptive chunker.

        Args:
            default_chunk_size: Default chunk size in seconds (for medium files)
            min_chunk_size: Minimum allowed chunk size in seconds
            max_chunk_size: Maximum allowed chunk size in seconds
        """
        self.default_chunk_size = default_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def calculate_chunk_size(
        self, duration: float, memory_budget: GPUMemoryBudget
    ) -> int:
        """Calculate optimal chunk size for a file.

        Strategy:
        1. Files ≤30s: Process whole file (no chunking overhead)
        2. Files 30-300s (5min): Use default_chunk_size (typically 10s)
        3. Files 300-1000s (5-16min): Adaptive 20-30s chunks
        4. Files >1000s (16min+): Adaptive 40-60s chunks to limit chunk count

        This ensures:
        - Short files: No unnecessary chunking
        - Typical files (<5min): User's preferred 10s chunks
        - Long files: Balanced between memory safety and efficiency

        Args:
            duration: File duration in seconds
            memory_budget: Current memory budget tracker (for future memory-based adaptation)

        Returns:
            Chunk size in seconds
        """
        # Very short files - no chunking needed
        if duration <= 30:
            return int(duration) + 1  # Slightly larger than file to process whole

        # Medium files (30s - 5min) - use default
        if duration <= 300:
            return self.default_chunk_size

        # Long files - adaptive sizing to reduce chunk count
        # Calculate number of chunks if using default size
        num_chunks_at_default = duration / self.default_chunk_size

        if num_chunks_at_default > 100:
            # Very long file (>16min at 10s chunks)
            # Target ~50 chunks maximum
            target_chunks = 50
            adaptive_size = int(duration / target_chunks)
            return min(self.max_chunk_size, max(self.min_chunk_size, adaptive_size))

        elif num_chunks_at_default > 50:
            # Long file (8-16min at 10s chunks)
            # Target ~30 chunks
            target_chunks = 30
            adaptive_size = int(duration / target_chunks)
            return min(30, max(self.min_chunk_size, adaptive_size))

        else:
            # Standard long file (5-8min)
            # Still use default chunk size
            return self.default_chunk_size

    def should_use_chunking(self, duration: float) -> bool:
        """Determine if file should be processed with chunking.

        Args:
            duration: File duration in seconds

        Returns:
            True if chunking should be used, False to process whole file
        """
        # Only chunk files longer than 30 seconds
        # This reduces overhead for short files
        return duration > 30

    def estimate_chunk_count(self, duration: float, memory_budget: GPUMemoryBudget) -> int:
        """Estimate number of chunks for a file.

        Args:
            duration: File duration in seconds
            memory_budget: Current memory budget tracker

        Returns:
            Estimated number of chunks
        """
        if not self.should_use_chunking(duration):
            return 1

        chunk_size = self.calculate_chunk_size(duration, memory_budget)
        return int(duration / chunk_size) + (1 if duration % chunk_size > 0 else 0)
