"""Checkpoint management for crash recovery in SAM-Audio batch processing.

This module provides checkpoint save/load functionality to enable recovery
from crashes or interruptions during long batch processing runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class CheckpointManager:
    """Manage checkpoint files for crash recovery.

    Checkpoints are stored as JSON files tracking processing progress at the
    file and chunk level. This enables resuming batch processing after crashes
    or interruptions without losing work.

    Checkpoint format (JSON):
    {
        "version": "1.0",
        "timestamp": "2026-01-09T10:30:00",
        "description": "remove background noise",
        "input_file": "/path/to/input.wav",
        "output_dir": "/path/to/output",
        "chunk_size": 10,
        "total_chunks": 30,
        "completed_chunks": [0, 1, 2, 3],
        "failed_chunks": [],
        "processing_params": {
            "rerank_candidates": 2,
            "predict_spans": false,
            "sample_rate": 16000
        },
        "status": "in_progress"  # or "completed", "failed"
    }

    Example:
        >>> mgr = CheckpointManager(Path(".checkpoints"))
        >>> # Check for existing checkpoint
        >>> checkpoint = mgr.load_checkpoint(Path("audio.wav"))
        >>> if checkpoint:
        >>>     pending = mgr.get_pending_chunks(checkpoint)
        >>>     # Process pending chunks
        >>>     for idx in pending:
        >>>         # ... process chunk ...
        >>>         mgr.mark_chunk_completed(checkpoint, idx)
        >>> mgr.remove_checkpoint(Path("audio.wav"))
    """

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, input_file: Path) -> Path:
        """Get checkpoint file path for an input file.

        Args:
            input_file: Input audio file path

        Returns:
            Path to checkpoint JSON file
        """
        # Sanitize filename: replace slashes and spaces with underscores
        safe_name = input_file.stem.replace("/", "_").replace(" ", "_")
        return self.checkpoint_dir / f"checkpoint_{safe_name}.json"

    def save_checkpoint(self, state: Dict) -> None:
        """Save checkpoint state to disk atomically.

        Uses atomic write pattern (temp file → rename) to prevent corruption
        if interrupted during write.

        Args:
            state: Checkpoint state dictionary
        """
        checkpoint_path = self.get_checkpoint_path(Path(state["input_file"]))
        state["timestamp"] = datetime.now().isoformat()

        # Atomic write: write to temp, then rename
        temp_path = checkpoint_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2)
            # Atomic rename (overwrites existing file)
            temp_path.replace(checkpoint_path)
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(self, input_file: Path) -> Optional[Dict]:
        """Load checkpoint for an input file if it exists.

        Args:
            input_file: Input audio file path

        Returns:
            Checkpoint state dictionary, or None if no checkpoint exists
        """
        checkpoint_path = self.get_checkpoint_path(input_file)
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)

            # Basic validation
            if "version" not in checkpoint or "status" not in checkpoint:
                print(f"Warning: Invalid checkpoint format in {checkpoint_path}")
                return None

            return checkpoint

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
            return None

    def remove_checkpoint(self, input_file: Path) -> None:
        """Remove checkpoint file (called on successful completion).

        Args:
            input_file: Input audio file path
        """
        checkpoint_path = self.get_checkpoint_path(input_file)
        checkpoint_path.unlink(missing_ok=True)

    def get_pending_chunks(self, checkpoint: Dict) -> List[int]:
        """Get list of chunks that still need processing.

        Args:
            checkpoint: Checkpoint state dictionary

        Returns:
            List of chunk indices that need processing
        """
        total = checkpoint.get("total_chunks", 0)
        completed = set(checkpoint.get("completed_chunks", []))

        # Return chunks that haven't been completed
        # (including failed ones to retry)
        return [i for i in range(total) if i not in completed]

    def mark_chunk_completed(self, checkpoint: Dict, chunk_idx: int) -> None:
        """Mark a chunk as completed and save checkpoint.

        Args:
            checkpoint: Checkpoint state dictionary
            chunk_idx: Chunk index to mark as completed
        """
        if "completed_chunks" not in checkpoint:
            checkpoint["completed_chunks"] = []

        if chunk_idx not in checkpoint["completed_chunks"]:
            checkpoint["completed_chunks"].append(chunk_idx)

        # Remove from failed list if present
        if "failed_chunks" in checkpoint and chunk_idx in checkpoint["failed_chunks"]:
            checkpoint["failed_chunks"].remove(chunk_idx)

        checkpoint["status"] = "in_progress"
        self.save_checkpoint(checkpoint)

    def mark_chunk_failed(self, checkpoint: Dict, chunk_idx: int, error: str) -> None:
        """Mark a chunk as failed and save checkpoint.

        Args:
            checkpoint: Checkpoint state dictionary
            chunk_idx: Chunk index to mark as failed
            error: Error message describing the failure
        """
        if "failed_chunks" not in checkpoint:
            checkpoint["failed_chunks"] = []

        if chunk_idx not in checkpoint["failed_chunks"]:
            checkpoint["failed_chunks"].append(chunk_idx)

        checkpoint["last_error"] = error
        self.save_checkpoint(checkpoint)

    def mark_completed(self, checkpoint: Dict) -> None:
        """Mark checkpoint as fully completed.

        Args:
            checkpoint: Checkpoint state dictionary
        """
        checkpoint["status"] = "completed"
        checkpoint["completion_time"] = datetime.now().isoformat()
        self.save_checkpoint(checkpoint)

    def mark_failed(self, checkpoint: Dict, error: str) -> None:
        """Mark checkpoint as failed.

        Args:
            checkpoint: Checkpoint state dictionary
            error: Error message describing the failure
        """
        checkpoint["status"] = "failed"
        checkpoint["error"] = error
        checkpoint["failure_time"] = datetime.now().isoformat()
        self.save_checkpoint(checkpoint)
