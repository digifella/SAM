"""Ultra-robust SAM-Audio runner optimized for WSL2 stability.

Key robustness features:
1. PyTorch memory fraction limiting (prevents over-allocation)
2. Incremental garbage collection checkpoints
3. Pre-flight memory checks before operations
4. Proper CUDA synchronization in all cleanup paths
5. Graceful degradation (fallback to simpler settings on OOM)
6. Detailed memory logging for debugging
"""

import os
from pathlib import Path
import argparse
import gc
import sys
import torch
import soundfile as sf

# Set PyTorch memory allocator settings BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio_utils import tensor_to_numpy, env_path, run_ffmpeg_convert

DEFAULT_INPUT_DIR = env_path("SAM_INPUT_DIR", Path("./audio_input"))
DEFAULT_OUTPUT_DIR = env_path("SAM_OUTPUT_DIR", Path("./audio_output"))
DEFAULT_CACHE_DIR = env_path("SAM_CACHE_DIR", Path("./audio_cache"))
DEFAULT_MODEL_DIR = env_path(
    "SAM_MODEL_DIR", Path.home() / "models" / "sam-audio-large-tv"
)
DEFAULT_FFMPEG_BIN = os.environ.get("SAM_FFMPEG_BIN", "ffmpeg")


def aggressive_cleanup():
    """Perform comprehensive memory cleanup with proper ordering."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # CRITICAL: wait for GPU ops to finish
        torch.cuda.empty_cache()
    gc.collect()


def get_memory_stats():
    """Get current GPU memory statistics in GB."""
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return {
        "allocated": f"{allocated:.2f}GB",
        "reserved": f"{reserved:.2f}GB",
        "total": f"{total:.2f}GB",
        "free": f"{total - allocated:.2f}GB"
    }


def log_memory(stage: str):
    """Log current memory state."""
    stats = get_memory_stats()
    if stats:
        print(f"[{stage}] GPU Memory: {stats['allocated']} allocated, "
              f"{stats['reserved']} reserved, {stats['free']} free of {stats['total']}")


def preflight_memory_check(required_gb: float = 2.0):
    """Check if we have enough free GPU memory before processing.

    Args:
        required_gb: Minimum free memory required in GB

    Returns:
        True if sufficient memory available, False otherwise
    """
    if not torch.cuda.is_available():
        return True

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    free = total - allocated

    if free < required_gb:
        print(f"WARNING: Insufficient GPU memory. Free: {free:.2f}GB, Required: {required_gb:.2f}GB")
        return False

    return True


def main():
    ap = argparse.ArgumentParser(description="Ultra-robust SAM-Audio separation for WSL2")
    ap.add_argument("--file", required=True, help="Input WAV filename or path")
    ap.add_argument(
        "--desc", required=True, help="Text description (e.g. 'young boy speaking')"
    )
    ap.add_argument("--rerank", type=int, default=1, help="Reranking candidates (>=1) [default: 1 for stability]")
    ap.add_argument(
        "--spans", action="store_true", help="Enable speech span prediction (uses more memory)"
    )
    ap.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory with SAM-Audio weights",
    )
    ap.add_argument(
        "--input_dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Base directory for audio inputs",
    )
    ap.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write separated tracks",
    )
    ap.add_argument(
        "--cache_dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for converted mono 16k WAVs",
    )
    ap.add_argument(
        "--ffmpeg_bin",
        default=DEFAULT_FFMPEG_BIN,
        help="ffmpeg executable to use for conversion",
    )
    ap.add_argument(
        "--convert",
        dest="convert",
        default=True,
        action="store_true",
        help="Convert to mono 16 kHz before inference (default: on)",
    )
    ap.add_argument(
        "--no-convert",
        dest="convert",
        action="store_false",
        help="Disable mono/16 kHz conversion",
    )
    ap.add_argument(
        "--convert_sr",
        type=int,
        default=16000,
        help="Sample rate for conversion",
    )
    ap.add_argument(
        "--force_reconvert",
        action="store_true",
        help="Ignore cached conversions",
    )
    ap.add_argument(
        "--memory_fraction",
        type=float,
        default=0.7,
        help="Max fraction of GPU memory to use (0.0-1.0) [default: 0.7 for WSL2]",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose memory logging",
    )
    args = ap.parse_args()

    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    input_path = Path(args.file)
    if not input_path.is_absolute():
        input_path = input_dir / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.convert:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAM-Audio Ultra-Robust Processing")
    print(f"{'='*70}")
    print(f"Input: {input_path.name}")
    print(f"Description: {args.desc}")
    print(f"Rerank: {args.rerank}, Spans: {args.spans}")

    # Configure device and memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        # Set memory fraction limit to prevent over-allocation
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
        print(f"GPU memory fraction limit: {args.memory_fraction}")
        log_memory("Initial")

    # Initial cleanup
    aggressive_cleanup()

    model = None
    processor = None
    batch = None
    result = None

    try:
        # Stage 1: Load processor (lightweight)
        if args.verbose:
            print("\n[Stage 1] Loading processor...")
        processor = SAMAudioProcessor.from_pretrained(args.model_dir)
        aggressive_cleanup()
        if args.verbose:
            log_memory("After processor load")

        # Stage 2: Load model (heavy)
        if args.verbose:
            print("\n[Stage 2] Loading model...")

        if not preflight_memory_check(required_gb=6.0):
            print("ERROR: Insufficient memory to load model. Try closing other applications.")
            sys.exit(1)

        model = SAMAudio.from_pretrained(args.model_dir).eval().to(device)
        aggressive_cleanup()
        if args.verbose:
            log_memory("After model load")

        # Stage 3: Audio conversion (if needed)
        proc_path = input_path
        if args.convert:
            proc_path = cache_dir / f"{input_path.stem}_mono{args.convert_sr}.wav"
            if args.force_reconvert or not proc_path.exists():
                if args.verbose:
                    print(f"\n[Stage 3] Converting audio...")
                print(f"Converting {input_path.name} -> {proc_path.name} ({args.convert_sr} Hz mono)")
                run_ffmpeg_convert(
                    input_path,
                    proc_path,
                    sample_rate=args.convert_sr,
                    channels=1,
                    ffmpeg_bin=args.ffmpeg_bin,
                )
            else:
                print(f"Using cached conversion -> {proc_path.name}")

        # Stage 4: Process audio
        if args.verbose:
            print("\n[Stage 4] Processing audio...")

        if not preflight_memory_check(required_gb=2.0):
            print("ERROR: Insufficient memory to process audio. Attempting cleanup...")
            aggressive_cleanup()
            if not preflight_memory_check(required_gb=2.0):
                print("ERROR: Still insufficient memory after cleanup.")
                sys.exit(1)

        batch = processor(audios=[str(proc_path)], descriptions=[args.desc]).to(device)
        aggressive_cleanup()
        if args.verbose:
            log_memory("After batch preparation")

        # Stage 5: Run inference with fallback strategy
        if args.verbose:
            print("\n[Stage 5] Running inference...")

        # Try progressively simpler settings if OOM occurs
        attempt_settings = [
            (max(1, args.rerank), bool(args.spans)),  # User-requested settings
            (1, False),                                 # Simplest fallback
        ]
        # Remove duplicate fallback if user already requested simplest settings
        if attempt_settings[0] == attempt_settings[1]:
            attempt_settings = [attempt_settings[0]]

        used_settings = None
        last_err = None

        for attempt_num, (rr, sp) in enumerate(attempt_settings, 1):
            try:
                if attempt_num > 1:
                    print(f"\nAttempt {attempt_num}: Trying with rerank={rr}, predict_spans={sp}")
                    aggressive_cleanup()
                    if args.verbose:
                        log_memory("After cleanup before retry")

                with torch.inference_mode():
                    result = model.separate(
                        batch,
                        predict_spans=sp,
                        reranking_candidates=max(1, rr)
                    )

                used_settings = (rr, sp)
                if args.verbose:
                    log_memory("After inference")
                break

            except RuntimeError as exc:
                last_err = exc
                if "out of memory" in str(exc).lower():
                    print(f"CUDA OOM detected on attempt {attempt_num}")
                    aggressive_cleanup()

                    if attempt_num == len(attempt_settings):
                        print("ERROR: Out of memory even with simplest settings")
                        raise

                    print("Retrying with simpler settings...")
                    continue
                else:
                    # Non-OOM error, don't retry
                    raise

        if result is None:
            raise RuntimeError(f"Failed to separate audio: {last_err}")

        # Stage 6: Save results
        if args.verbose:
            print("\n[Stage 6] Saving results...")

        sr = int(processor.audio_sampling_rate)
        out_base = input_path.stem
        target_path = output_dir / f"{out_base}_target.wav"
        residual_path = output_dir / f"{out_base}_residual.wav"

        # Convert tensors to numpy and free GPU memory immediately
        target_numpy = tensor_to_numpy(result.target)
        residual_numpy = tensor_to_numpy(result.residual)

        # Clear result from GPU before writing files
        del result
        aggressive_cleanup()

        # Write files
        sf.write(target_path, target_numpy, sr, subtype="PCM_16")
        sf.write(residual_path, residual_numpy, sr, subtype="PCM_16")

        print(f"\n{'='*70}")
        print(f"SUCCESS! Output files:")
        print(f"  Target:   {target_path}")
        print(f"  Residual: {residual_path}")
        print(f"  Sample rate: {sr} Hz")

        if used_settings and used_settings != (max(1, args.rerank), bool(args.spans)):
            print(f"\nNote: Automatically fell back to rerank={used_settings[0]}, "
                  f"predict_spans={used_settings[1]} due to memory constraints")

        if args.verbose:
            log_memory("Final")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        if args.verbose:
            log_memory("Error state")
        raise

    finally:
        # Comprehensive cleanup - free everything
        if args.verbose:
            print("\n[Cleanup] Releasing all resources...")

        if batch is not None:
            del batch
        if result is not None:
            del result
        if model is not None:
            del model
        if processor is not None:
            del processor

        aggressive_cleanup()

        if args.verbose and torch.cuda.is_available():
            log_memory("After final cleanup")


if __name__ == "__main__":
    main()
