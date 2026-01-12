import os
from pathlib import Path
import argparse
import gc
import torch
import soundfile as sf

# CRITICAL FIX: Set PyTorch memory allocator config BEFORE importing model
# This prevents memory fragmentation that causes WSL2 crashes
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,expandable_segments:True')

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio_utils import tensor_to_numpy, env_path, run_ffmpeg_convert

DEFAULT_INPUT_DIR = env_path("SAM_INPUT_DIR", Path("./audio_input"))
DEFAULT_OUTPUT_DIR = env_path("SAM_OUTPUT_DIR", Path("./audio_output"))
DEFAULT_CACHE_DIR = env_path("SAM_CACHE_DIR", Path("./audio_cache"))
DEFAULT_MODEL_DIR = env_path(
    "SAM_MODEL_DIR", Path.home() / "models" / "sam-audio-large-tv"
)
DEFAULT_FFMPEG_BIN = os.environ.get("SAM_FFMPEG_BIN", "ffmpeg")

def main():
    ap = argparse.ArgumentParser(description="Run SAM-Audio separation")
    ap.add_argument("--file", required=True, help="Input WAV filename or path")
    ap.add_argument(
        "--desc", required=True, help="Text description (e.g. 'young boy speaking')"
    )
    ap.add_argument("--rerank", type=int, default=4, help="Reranking candidates (>=1)")
    ap.add_argument(
        "--spans", action="store_true", help="Enable speech span prediction"
    )
    ap.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory with SAM-Audio weights",
    )
    ap.add_argument(
        "--input_dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Base directory for audio inputs (overridden by absolute paths)",
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
        help="Sample rate to use when conversion is enabled",
    )
    ap.add_argument(
        "--force_reconvert",
        action="store_true",
        help="Ignore cached conversions and regenerate",
    )
    args = ap.parse_args()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ROBUSTNESS FIX: Limit GPU memory fraction to prevent WSL2 over-allocation
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.7, device=0)

    model = SAMAudio.from_pretrained(args.model_dir).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.model_dir)

    proc_path = input_path
    if args.convert:
        proc_path = cache_dir / f"{input_path.stem}_mono{args.convert_sr}.wav"
        if args.force_reconvert or not proc_path.exists():
            print(
                f"Converting {input_path.name} -> {proc_path.name} ({args.convert_sr} Hz mono)"
            )
            run_ffmpeg_convert(
                input_path,
                proc_path,
                sample_rate=args.convert_sr,
                channels=1,
                ffmpeg_bin=args.ffmpeg_bin,
            )
        else:
            print(f"Using cached conversion -> {proc_path.name}")

    batch = None
    result = None
    try:
        batch = processor(audios=[str(proc_path)], descriptions=[args.desc]).to(device)

        attempt_settings = [
            (max(1, args.rerank), bool(args.spans)),
            (1, False) if (args.rerank != 1 or args.spans) else None,
        ]
        attempt_settings = [x for x in attempt_settings if x is not None]
        used_settings = None
        last_err = None
        for rr, sp in attempt_settings:
            try:
                with torch.inference_mode():
                    result = model.separate(
                        batch,
                        predict_spans=sp,
                        reranking_candidates=max(1, rr)
                    )
                used_settings = (rr, sp)
                break
            except RuntimeError as exc:
                last_err = exc
                if "out of memory" in str(exc).lower():
                    print("CUDA OOM detected; retrying with rerank=1 and predict_spans=False")
                    if device == "cuda":
                        gc.collect()
                        torch.cuda.synchronize()  # Wait for all GPU operations to complete
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                raise

        if result is None:
            raise RuntimeError(f"Failed to separate audio: {last_err}")

        sr = int(processor.audio_sampling_rate)

        out_base = input_path.stem
        target_path = output_dir / f"{out_base}_target.wav"
        residual_path = output_dir / f"{out_base}_residual.wav"

        sf.write(target_path, tensor_to_numpy(result.target), sr, subtype="PCM_16")
        sf.write(residual_path, tensor_to_numpy(result.residual), sr, subtype="PCM_16")

        print(f"Wrote:\n  {target_path}\n  {residual_path}\n@ {sr} Hz")
        if used_settings and used_settings != (max(1, args.rerank), bool(args.spans)):
            print(
                f"Note: automatically fell back to rerank={used_settings[0]} predict_spans={used_settings[1]}"
            )

    finally:
        # CRITICAL CLEANUP ORDER: Prevents memory leaks and WSL2 crashes
        # 1. Delete Python objects first
        if batch is not None:
            del batch
        if result is not None:
            del result

        # 2. Run GC to free Python references to CUDA memory
        gc.collect()

        # 3. CRITICAL: Synchronize CUDA to ensure all ops finish
        if device == "cuda":
            torch.cuda.synchronize()  # Wait for GPU operations
            torch.cuda.empty_cache()  # Free cached memory
            gc.collect()              # Final Python GC pass

if __name__ == "__main__":
    main()
