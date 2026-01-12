#!/usr/bin/env python3
"""Robust SAM-Audio batch processor with checkpoint recovery and memory management.

This script provides a production-ready batch processing system for SAM-Audio
with the following features:
- Aggressive GPU memory cleanup (fixes WSL2 crashes)
- Checkpoint-based crash recovery
- Adaptive chunking for long files
- Pre-flight system validation
- Non-interactive mode support
- Comprehensive error handling

Key fixes for WSL2 stability:
1. torch.cuda.synchronize() before empty_cache() (critical!)
2. CUDA config: expandable_segments:True,max_split_size_mb:2048
3. Reduced GPU memory fraction (0.55 vs 0.70)
4. Try/finally cleanup patterns
"""

import os
# CRITICAL: Must be set BEFORE importing torch to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:2048"

from pathlib import Path
import argparse
import sys
import shutil
import subprocess

import torch
import soundfile as sf

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio_utils import tensor_to_numpy, run_ffmpeg_convert, env_path
from sam_audio_utils.memory import GPUMemoryBudget, estimate_chunk_memory
from sam_audio_utils.checkpoint import CheckpointManager
from sam_audio_utils.chunking import AdaptiveChunker


# Default paths from environment or fallback values
DEFAULT_MODEL_DIR = env_path("SAM_MODEL_DIR", Path.home() / "models" / "sam-audio-large-tv")
DEFAULT_FFMPEG_BIN = os.environ.get("SAM_FFMPEG_BIN", "ffmpeg")


def run_cmd(cmd):
    """Run subprocess command and raise on error."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n  {' '.join(cmd)}\n\n{proc.stderr[-3000:]}")
    return proc


def ffmpeg_split_wav(in_wav: Path, out_dir: Path, chunk_s: int, ffmpeg_bin: str):
    """Split WAV into fixed-length chunks using ffmpeg segment muxer.

    Args:
        in_wav: Input WAV file
        out_dir: Output directory for chunks
        chunk_s: Chunk length in seconds
        ffmpeg_bin: Path to ffmpeg binary

    Produces: chunk_000000.wav, chunk_000001.wav, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "chunk_%06d.wav")
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(in_wav),
        "-f", "segment",
        "-segment_time", str(chunk_s),
        "-reset_timestamps", "1",
        out_pattern
    ]
    run_cmd(cmd)


def concat_wavs_with_ffmpeg(wavs, out_path: Path, ffmpeg_bin: str):
    """Concatenate WAVs via ffmpeg concat demuxer.

    Args:
        wavs: List of WAV file paths to concatenate
        out_path: Output file path
        ffmpeg_bin: Path to ffmpeg binary
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.parent / f".concat_{out_path.stem}.txt"

    # Create concat list file
    lines = []
    for w in wavs:
        # Quote single quotes safely for ffmpeg
        p = str(w).replace("'", "'\\''")
        lines.append(f"file '{p}'")
    list_file.write_text("\n".join(lines) + "\n")

    # Concatenate
    cmd = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)]
    run_cmd(cmd)

    # Cleanup list file
    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass


class PreflightChecker:
    """Validate system state before batch processing."""

    def __init__(self, input_dir: Path, output_dir: Path, work_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.work_dir = work_dir
        self.errors = []
        self.warnings = []

    def check_directories(self) -> bool:
        """Verify directory access and permissions."""
        if not self.input_dir.exists():
            self.errors.append(f"Input directory not found: {self.input_dir}")
            return False

        # Output and work dirs must be writable
        for dir_path, name in [(self.output_dir, "output"), (self.work_dir, "work")]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                test_file = dir_path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                self.errors.append(f"Cannot write to {name} directory {dir_path}: {e}")
                return False

        return True

    def check_disk_space(self, required_gb: float = 10.0) -> bool:
        """Check available disk space."""
        for dir_path, name in [(self.output_dir, "output"), (self.work_dir, "work")]:
            try:
                stat = shutil.disk_usage(dir_path)
                available_gb = stat.free / (1024**3)

                if available_gb < required_gb:
                    self.warnings.append(
                        f"Low disk space in {name} directory: {available_gb:.1f}GB "
                        f"available (recommended: {required_gb}GB)"
                    )
            except Exception as e:
                self.warnings.append(f"Could not check disk space for {dir_path}: {e}")

        return True  # Warning only, not fatal

    def check_gpu(self) -> bool:
        """Validate GPU availability and memory."""
        if not torch.cuda.is_available():
            self.errors.append("CUDA not available. This script requires GPU.")
            return False

        try:
            device_props = torch.cuda.get_device_properties(0)
            total_gb = device_props.total_memory / (1024**3)

            if total_gb < 8:
                self.warnings.append(
                    f"GPU has only {total_gb:.1f}GB memory. "
                    "Recommended: 16GB+ (48GB optimal)"
                )

            print(f"GPU: {device_props.name} with {total_gb:.1f}GB memory")

        except Exception as e:
            self.errors.append(f"GPU check failed: {e}")
            return False

        return True

    def check_ffmpeg(self, ffmpeg_bin: str) -> bool:
        """Verify ffmpeg is available."""
        if shutil.which(ffmpeg_bin) is None:
            self.errors.append(
                f"ffmpeg binary '{ffmpeg_bin}' not found. "
                "Install ffmpeg or set --ffmpeg_bin"
            )
            return False

        return True

    def check_audio_files(self) -> bool:
        """Scan for audio files and estimate durations."""
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        files = [f for f in self.input_dir.rglob("*")
                 if f.is_file() and f.suffix.lower() in audio_exts]

        if not files:
            self.errors.append(f"No audio files found in {self.input_dir}")
            return False

        print(f"Found {len(files)} audio file(s) to process")

        # Quick duration check on first few files
        try:
            total_duration = 0
            for f in files[:5]:
                try:
                    info = sf.info(str(f))
                    total_duration += info.duration
                except:
                    pass

            if total_duration > 0:
                avg_duration = total_duration / min(len(files), 5)
                est_total = avg_duration * len(files)
                print(f"Estimated total duration: {est_total/60:.1f} minutes")

        except ImportError:
            pass

        return True

    def run_all_checks(self, ffmpeg_bin: str) -> bool:
        """Run all pre-flight checks."""
        print("=" * 70)
        print("PRE-FLIGHT CHECKS")
        print("=" * 70)

        checks = [
            ("Directories", self.check_directories),
            ("Disk space", self.check_disk_space),
            ("GPU", self.check_gpu),
            ("ffmpeg", lambda: self.check_ffmpeg(ffmpeg_bin)),
            ("Audio files", self.check_audio_files),
        ]

        all_passed = True
        for name, check_fn in checks:
            print(f"\nChecking {name}...", end=" ")
            try:
                result = check_fn()
                print("✓" if result else "✗")
                all_passed = all_passed and result
            except Exception as e:
                print(f"✗ (Exception: {e})")
                self.errors.append(f"{name} check failed: {e}")
                all_passed = False

        # Display all errors and warnings
        if self.errors:
            print("\n" + "=" * 70)
            print("ERRORS:")
            for err in self.errors:
                print(f"  ✗ {err}")

        if self.warnings:
            print("\n" + "=" * 70)
            print("WARNINGS:")
            for warn in self.warnings:
                print(f"  ! {warn}")

        if all_passed:
            print("\n" + "=" * 70)
            print("All pre-flight checks passed!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("Pre-flight checks FAILED. Please resolve errors above.")
            print("=" * 70)

        return all_passed


def process_file_whole(
    converted_wav: Path,
    description: str,
    model,
    processor,
    output_dir: Path,
    base_name: str,
    device: str,
    memory_budget: GPUMemoryBudget,
    args,
) -> bool:
    """Process entire file without chunking.

    Args:
        converted_wav: Converted mono 16kHz WAV file
        description: Target audio description
        model: SAMAudio model
        processor: SAMAudioProcessor
        output_dir: Output directory
        base_name: Base name for output files
        device: Device string ("cuda" or "cpu")
        memory_budget: Memory budget tracker
        args: Command-line arguments

    Returns:
        True if successful, False otherwise
    """
    batch = None
    result = None
    try:
        batch = processor(audios=[str(converted_wav)], descriptions=[description]).to(device)

        # Try with requested settings, fall back on OOM
        attempts = [
            (args.rerank_candidates, args.predict_spans),
            (1, False),  # Safest fallback
        ]

        success = False
        for rerank, spans in attempts:
            try:
                with torch.inference_mode():
                    result = model.separate(
                        batch,
                        predict_spans=spans,
                        reranking_candidates=max(1, rerank)
                    )
                success = True
                if (rerank, spans) != attempts[0]:
                    print(f"  (Used fallback settings: rerank={rerank}, spans={spans})")
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM, trying safer settings...")
                    memory_budget.aggressive_cleanup()
                    continue
                else:
                    raise

        if not success:
            print("ERROR: Processing failed (OOM)")
            return False

        # Save outputs
        sr = int(processor.audio_sampling_rate)
        target_path = output_dir / f"{base_name}_target.wav"
        resid_path = output_dir / f"{base_name}_residual.wav"

        sf.write(target_path, tensor_to_numpy(result.target), sr, subtype="PCM_16")
        sf.write(resid_path, tensor_to_numpy(result.residual), sr, subtype="PCM_16")

        print(f"✓ Completed: {target_path.name} and {resid_path.name}")
        return True

    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        return False

    finally:
        # CRITICAL: Always cleanup, even on exception
        del batch, result
        memory_budget.aggressive_cleanup()


def process_file_chunked(
    converted_wav: Path,
    description: str,
    model,
    processor,
    output_dir: Path,
    base_name: str,
    work_dir: Path,
    device: str,
    memory_budget: GPUMemoryBudget,
    checkpoint: dict,
    checkpoint_mgr: CheckpointManager,
    args,
) -> bool:
    """Process file in chunks with checkpoint recovery.

    Args:
        converted_wav: Converted mono 16kHz WAV file
        description: Target audio description
        model: SAMAudio model
        processor: SAMAudioProcessor
        output_dir: Output directory
        base_name: Base name for output files
        work_dir: Working directory for chunks
        device: Device string ("cuda" or "cpu")
        memory_budget: Memory budget tracker
        checkpoint: Checkpoint state dictionary
        checkpoint_mgr: Checkpoint manager
        args: Command-line arguments

    Returns:
        True if successful, False otherwise
    """
    # Create chunk directories
    chunks_dir = work_dir / "chunks"
    out_target_dir = work_dir / "target_chunks"
    out_resid_dir = work_dir / "residual_chunks"

    for d in (chunks_dir, out_target_dir, out_resid_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Split into chunks (if not already done)
    chunk_files = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunk_files:
        print(f"Splitting into {checkpoint['chunk_size']}s chunks...")
        ffmpeg_split_wav(
            converted_wav,
            chunks_dir,
            chunk_size=checkpoint['chunk_size'],
            ffmpeg_bin=args.ffmpeg_bin
        )
        chunk_files = sorted(chunks_dir.glob("chunk_*.wav"))

    if not chunk_files:
        print("ERROR: No chunks produced")
        return False

    # Get pending chunks
    pending_chunks = checkpoint_mgr.get_pending_chunks(checkpoint)
    completed_count = len(checkpoint.get("completed_chunks", []))

    print(f"Processing {len(pending_chunks)} chunk(s) ({completed_count} already completed)...")

    # Process each pending chunk
    for chunk_idx in pending_chunks:
        if chunk_idx >= len(chunk_files):
            print(f"WARNING: Chunk {chunk_idx} out of range, skipping")
            continue

        chunk_file = chunk_files[chunk_idx]
        print(f"  Chunk {chunk_idx+1}/{len(chunk_files)}: {chunk_file.name}")

        # Check memory budget before processing
        estimated_mem = estimate_chunk_memory(checkpoint['chunk_size'], args.sample_rate)
        if not memory_budget.can_process_chunk(estimated_mem):
            print(f"    WARNING: Insufficient GPU memory, performing cleanup...")
            memory_budget.aggressive_cleanup()

            # Recheck
            if not memory_budget.can_process_chunk(estimated_mem):
                print(f"    ERROR: Still insufficient memory after cleanup")
                checkpoint_mgr.mark_chunk_failed(checkpoint, chunk_idx, "Insufficient GPU memory")
                continue

        # Process chunk with aggressive cleanup
        batch = None
        result = None
        try:
            batch = processor(audios=[str(chunk_file)], descriptions=[description]).to(device)

            # Try with requested settings, fall back to safer settings on OOM
            attempts = [
                (args.rerank_candidates, args.predict_spans),
                (1, False),  # Safest fallback
            ]

            success = False
            for rerank, spans in attempts:
                try:
                    with torch.inference_mode():
                        result = model.separate(
                            batch,
                            predict_spans=spans,
                            reranking_candidates=max(1, rerank)
                        )
                    success = True
                    if (rerank, spans) != attempts[0]:
                        print(f"    (Used fallback settings: rerank={rerank}, spans={spans})")
                    break

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"    OOM with rerank={rerank}, spans={spans}, trying safer settings...")
                        memory_budget.aggressive_cleanup()
                        continue
                    else:
                        raise

            if not success:
                checkpoint_mgr.mark_chunk_failed(checkpoint, chunk_idx, "All processing attempts failed (OOM)")
                continue

            # Save chunk outputs
            sr = int(processor.audio_sampling_rate)
            target_path = out_target_dir / f"{chunk_file.stem}_target.wav"
            resid_path = out_resid_dir / f"{chunk_file.stem}_residual.wav"

            sf.write(target_path, tensor_to_numpy(result.target), sr, subtype="PCM_16")
            sf.write(resid_path, tensor_to_numpy(result.residual), sr, subtype="PCM_16")

            # Mark chunk completed
            checkpoint_mgr.mark_chunk_completed(checkpoint, chunk_idx)
            print(f"    ✓ Chunk {chunk_idx+1} completed")

        except Exception as e:
            print(f"    ✗ Chunk {chunk_idx+1} failed: {e}")
            checkpoint_mgr.mark_chunk_failed(checkpoint, chunk_idx, str(e))
            continue

        finally:
            # CRITICAL: Always cleanup after each chunk
            del batch, result
            memory_budget.aggressive_cleanup()

    # Collect all successfully processed chunks (including previously completed)
    all_target_paths = sorted(out_target_dir.glob("chunk_*_target.wav"))
    all_resid_paths = sorted(out_resid_dir.glob("chunk_*_residual.wav"))

    if len(all_target_paths) != len(chunk_files):
        print(f"ERROR: Only {len(all_target_paths)}/{len(chunk_files)} chunks succeeded")
        return False

    # Concatenate outputs
    final_target = output_dir / f"{base_name}_target.wav"
    final_resid = output_dir / f"{base_name}_residual.wav"

    print("Concatenating target chunks...")
    concat_wavs_with_ffmpeg(all_target_paths, final_target, args.ffmpeg_bin)

    print("Concatenating residual chunks...")
    concat_wavs_with_ffmpeg(all_resid_paths, final_resid, args.ffmpeg_bin)

    print(f"✓ Completed: {final_target.name} and {final_resid.name}")
    return True


def process_file(
    input_file: Path,
    description: str,
    model,
    processor,
    output_dir: Path,
    work_dir: Path,
    device: str,
    memory_budget: GPUMemoryBudget,
    chunker: AdaptiveChunker,
    checkpoint_mgr: CheckpointManager,
    args,
) -> bool:
    """Process a single audio file with checkpoint recovery.

    Args:
        input_file: Input audio file
        description: Target audio description
        model: SAMAudio model
        processor: SAMAudioProcessor
        output_dir: Output directory
        work_dir: Working directory
        device: Device string ("cuda" or "cpu")
        memory_budget: Memory budget tracker
        chunker: Adaptive chunker
        checkpoint_mgr: Checkpoint manager
        args: Command-line arguments

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")

    # Check for existing checkpoint
    checkpoint = None
    if not args.no_checkpoint:
        checkpoint = checkpoint_mgr.load_checkpoint(input_file)
        if checkpoint:
            print(f"Found existing checkpoint from {checkpoint.get('timestamp', 'unknown')}")
            print(f"  Completed chunks: {len(checkpoint.get('completed_chunks', []))} / "
                  f"{checkpoint.get('total_chunks', 0)}")

            if args.resume:
                print("Resuming from checkpoint...")
            else:
                response = input("Resume from checkpoint? (Y/n): ").strip().lower()
                if response not in ("", "y", "yes"):
                    print("Starting fresh...")
                    checkpoint_mgr.remove_checkpoint(input_file)
                    checkpoint = None

    # Convert to mono 16kHz WAV
    file_work = work_dir / input_file.stem
    file_work.mkdir(parents=True, exist_ok=True)

    converted_wav = file_work / f"{input_file.stem}_mono{args.sample_rate}.wav"

    if not converted_wav.exists():
        print("Converting to mono 16kHz...")
        run_ffmpeg_convert(
            input_file,
            converted_wav,
            sample_rate=args.sample_rate,
            channels=1,
            ffmpeg_bin=args.ffmpeg_bin,
        )
    else:
        print("Using existing converted file...")

    # Get file duration
    info = sf.info(str(converted_wav))
    duration = float(info.duration)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

    # Determine if chunking is needed
    use_chunking = chunker.should_use_chunking(duration)
    chunk_size = chunker.calculate_chunk_size(duration, memory_budget) if use_chunking else int(duration) + 1

    print(f"Processing mode: {'Chunked' if use_chunking else 'Whole file'}")
    if use_chunking:
        print(f"Chunk size: {chunk_size}s")

    # Initialize or resume checkpoint
    if checkpoint is None and not args.no_checkpoint:
        num_chunks = int(duration / chunk_size) + (1 if duration % chunk_size > 0 else 0)
        checkpoint = {
            "version": "1.0",
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "description": description,
            "chunk_size": chunk_size,
            "total_chunks": num_chunks if use_chunking else 1,
            "completed_chunks": [],
            "failed_chunks": [],
            "processing_params": {
                "rerank_candidates": args.rerank_candidates,
                "predict_spans": args.predict_spans,
                "sample_rate": args.sample_rate,
            },
            "status": "in_progress",
        }
        checkpoint_mgr.save_checkpoint(checkpoint)

    # Process whole file or chunks
    success = False
    try:
        if use_chunking:
            success = process_file_chunked(
                converted_wav=converted_wav,
                description=description,
                model=model,
                processor=processor,
                output_dir=output_dir,
                base_name=input_file.stem,
                work_dir=file_work,
                device=device,
                memory_budget=memory_budget,
                checkpoint=checkpoint,
                checkpoint_mgr=checkpoint_mgr,
                args=args,
            )
        else:
            success = process_file_whole(
                converted_wav=converted_wav,
                description=description,
                model=model,
                processor=processor,
                output_dir=output_dir,
                base_name=input_file.stem,
                device=device,
                memory_budget=memory_budget,
                args=args,
            )

        if success and not args.no_checkpoint:
            checkpoint_mgr.remove_checkpoint(input_file)

    finally:
        # Cleanup work directory unless --keep_work
        if not args.keep_work:
            shutil.rmtree(file_work, ignore_errors=True)

        # CRITICAL: Aggressive memory cleanup between files
        memory_budget.aggressive_cleanup()

    return success


def create_argument_parser():
    """Create argument parser with simple, intuitive options."""
    ap = argparse.ArgumentParser(
        description="Robust SAM-Audio batch processor with checkpoint recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (will prompt for description):
  python run_sam_audio_batch_robust.py --input_dir ./audio_input --output_dir ./audio_output

  # Non-interactive with description:
  python run_sam_audio_batch_robust.py \\
    --input_dir ./audio_input \\
    --output_dir ./audio_output \\
    --description "female speaker, remove background music"

  # Resume after crash:
  python run_sam_audio_batch_robust.py --input_dir ./audio_input --output_dir ./audio_output --resume

  # Advanced: custom chunk size and memory settings:
  python run_sam_audio_batch_robust.py \\
    --input_dir ./audio_input \\
    --output_dir ./audio_output \\
    --chunk_size 10 \\
    --gpu_mem_frac 0.55
        """
    )

    # Required arguments
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing audio files to process")
    ap.add_argument("--output_dir", required=True,
                    help="Directory for separated audio outputs (_target.wav and _residual.wav)")

    # Optional core arguments
    ap.add_argument("--description",
                    help="Target audio description (e.g., 'woman speaking'). "
                         "If not provided, will prompt interactively.")
    ap.add_argument("--work_dir", default="./audio_work_robust",
                    help="Working directory for temporary files (default: ./audio_work_robust)")

    # Checkpoint arguments
    ap.add_argument("--checkpoint_dir", default="./.checkpoints",
                    help="Directory for checkpoint files (default: ./.checkpoints)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing checkpoints if found")
    ap.add_argument("--no_checkpoint", action="store_true",
                    help="Disable checkpoint system (not recommended)")

    # Processing parameters
    ap.add_argument("--chunk_size", type=int, default=10,
                    help="Default chunk size in seconds for long files (default: 10)")
    ap.add_argument("--rerank_candidates", type=int, default=2,
                    help="Number of reranking candidates (1-4, default: 2)")
    ap.add_argument("--predict_spans", action="store_true",
                    help="Enable predict_spans (increases memory usage)")

    # Audio conversion
    ap.add_argument("--sample_rate", type=int, default=16000,
                    help="Sample rate for conversion (default: 16000)")

    # GPU settings
    ap.add_argument("--gpu_mem_frac", type=float, default=0.55,
                    help="GPU memory fraction limit (0-1, default: 0.55 for WSL2)")

    # Model and tools
    ap.add_argument("--model_dir",
                    default=str(DEFAULT_MODEL_DIR),
                    help="Path to SAM-Audio model directory")
    ap.add_argument("--ffmpeg_bin", default=DEFAULT_FFMPEG_BIN,
                    help="Path to ffmpeg binary (default: ffmpeg)")

    # Utility options
    ap.add_argument("--clean_work", action="store_true",
                    help="Clean work directory before starting")
    ap.add_argument("--keep_work", action="store_true",
                    help="Keep work directory after completion (for debugging)")
    ap.add_argument("--offline", action="store_true",
                    help="Force offline mode (no HuggingFace downloads)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Scan files and show what would be processed without running")

    return ap


def main():
    """Main entry point."""
    ap = create_argument_parser()
    args = ap.parse_args()

    # Offline mode
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Parse paths
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()

    # Pre-flight checks
    checker = PreflightChecker(input_dir, output_dir, work_dir)
    if not checker.run_all_checks(args.ffmpeg_bin):
        print("\nAborting due to pre-flight check failures.")
        return 1

    # Clean work directory if requested
    if args.clean_work and work_dir.exists():
        print(f"Cleaning work directory: {work_dir}")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Get description (prompt if not provided)
    description = args.description
    if not description:
        description = input("\nTarget audio description (e.g., 'woman speaking, remove cicadas'): ").strip()
        while not description:
            description = input("Description cannot be empty. Please provide: ").strip()

    print(f"\nDescription: '{description}'")

    # Find audio files
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = sorted([
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in audio_exts
    ])

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return 1

    print(f"\nFound {len(audio_files)} file(s) to process")

    # Dry run mode
    if args.dry_run:
        print("\nDRY RUN - Would process:")
        for i, f in enumerate(audio_files, 1):
            print(f"  {i}. {f.relative_to(input_dir)}")
        return 0

    # Confirm start
    if not args.resume:
        response = input(f"\nStart processing {len(audio_files)} file(s)? (Y/n): ").strip().lower()
        if response not in ("", "y", "yes"):
            print("Aborted.")
            return 0

    # Initialize CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac, device=0)
        print(f"\nGPU memory fraction: {args.gpu_mem_frac}")

    # Load model
    print(f"Loading model from {model_dir}...")
    model = SAMAudio.from_pretrained(str(model_dir)).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(str(model_dir))
    print("Model loaded.")

    # Initialize systems
    memory_budget = GPUMemoryBudget(device, reserve_fraction=args.gpu_mem_frac)
    memory_budget.establish_baseline()

    chunker = AdaptiveChunker(
        default_chunk_size=args.chunk_size,
        min_chunk_size=5,
        max_chunk_size=60
    )

    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # Process files
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING START")
    print(f"{'='*70}")

    success_count = 0
    failed_files = []

    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n[{idx}/{len(audio_files)}]")

        try:
            success = process_file(
                input_file=audio_file,
                description=description,
                model=model,
                processor=processor,
                output_dir=output_dir,
                work_dir=work_dir,
                device=device,
                memory_budget=memory_budget,
                chunker=chunker,
                checkpoint_mgr=checkpoint_mgr,
                args=args,
            )

            if success:
                success_count += 1
            else:
                failed_files.append(audio_file)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            print(f"Progress: {success_count}/{idx} files completed")
            print("Resume with --resume flag to continue from checkpoints.")
            return 130

        except Exception as e:
            print(f"ERROR: Unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(audio_file)

    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {success_count}/{len(audio_files)}")

    if failed_files:
        print(f"Failed files:")
        for f in failed_files:
            print(f"  - {f.relative_to(input_dir)}")
        return 1
    else:
        print("All files processed successfully!")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
