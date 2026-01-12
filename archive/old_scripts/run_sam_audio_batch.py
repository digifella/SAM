import os
# Must be set BEFORE importing torch to reduce CUDA fragmentation issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import argparse
import sys
import gc
import shutil
import subprocess

import torch
import soundfile as sf

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio_utils import tensor_to_numpy, run_ffmpeg_convert, env_path

DEFAULT_INPUT_DIR = env_path("SAM_INPUT_DIR", Path("./audio_input"))
DEFAULT_OUTPUT_DIR = env_path("SAM_OUTPUT_DIR", Path("./audio_output"))
DEFAULT_CACHE_DIR = env_path("SAM_CACHE_DIR", Path("./audio_cache"))
DEFAULT_MODEL_DIR = env_path("SAM_MODEL_DIR", Path.home() / "models" / "sam-audio-large-tv")
DEFAULT_FFMPEG_BIN = os.environ.get("SAM_FFMPEG_BIN", "ffmpeg")

def prompt_str(prompt, default=None):
    if default is None:
        s = input(f"{prompt}: ").strip()
        while not s:
            s = input(f"{prompt} (cannot be empty): ").strip()
        return s
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default

def prompt_int(prompt, default, min_value=1):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
            if v < min_value:
                print(f"  Must be >= {min_value}. Try again.")
                continue
            return v
        except ValueError:
            print("  Please enter an integer.")

def prompt_float(prompt, default, min_value=0.05, max_value=1.0):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            v = float(s)
            if v < min_value or v > max_value:
                print(f"  Must be between {min_value} and {max_value}. Try again.")
                continue
            return v
        except ValueError:
            print("  Please enter a number.")

def prompt_bool(prompt, default):
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} ({d}): ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("  Please answer y or n.")

def safe_empty_cuda(device):
    """Safely clear CUDA cache with proper synchronization.

    CRITICAL: Must call synchronize() before empty_cache() to ensure all
    GPU operations complete. Without this, pending operations hold memory
    and cause fragmentation, leading to OOM crashes especially in WSL2.
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
        torch.cuda.empty_cache()
    gc.collect()  # Final cleanup of Python objects

def audio_duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Interactive batch SAM-Audio separation (F: drive I/O, OOM-hardened)")
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--pattern", default="*.wav", help="Glob pattern to match inputs")
    ap.add_argument("--offline", action="store_true", help="Force HF/Transformers offline mode (cache only)")
    ap.add_argument("--default_rerank", type=int, default=2, help="Default reranking candidates (>=1). 2 is safer than 4.")
    ap.add_argument("--default_spans_on", action="store_true", help="Default predict_spans ON (can increase memory)")
    ap.add_argument("--gpu_mem_frac", type=float, default=0.70, help="CUDA memory fraction cap (0-1)")
    ap.add_argument("--auto_convert_16k", action="store_true", help="Auto-convert each input to mono 16k WAV before processing (recommended)")
    ap.add_argument("--convert_sr", type=int, default=16000, help="Sample rate for conversion (default 16000)")
    ap.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing input files")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for separated outputs")
    ap.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR), help="Directory for converted WAV cache")
    ap.add_argument("--ffmpeg_bin", default=DEFAULT_FFMPEG_BIN, help="ffmpeg binary to use for conversions")
    ap.add_argument("--clear_cache", action="store_true", help="Delete cache_dir contents before running")
    ap.add_argument("--auto_chunk", action="store_true", help="Automatically run chunked pipeline for long files")
    ap.add_argument("--chunk_threshold_s", type=int, default=180, help="Duration (seconds) above which chunked mode is used")
    ap.add_argument("--chunk_len", type=int, default=20, help="Chunk length in seconds when chunking is triggered")
    ap.add_argument("--chunk_script", default=str(Path(__file__).with_name("run_sam_audio_batch_chunked.py")), help="Path to chunked runner")
    args = ap.parse_args()

    # Offline-first: avoid hub chatter and accidental network usage once cached
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_work_root = cache_dir.parent / "chunk_work"

    def launch_chunk_runner(in_path: Path, desc_text: str, rerank_value: int, spans_on: bool, reason: str) -> bool:
        print(reason)
        chunk_cmd = [
            sys.executable,
            args.chunk_script,
            "--model_dir",
            args.model_dir,
            "--pattern",
            in_path.name,
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--work_dir",
            str(chunk_work_root),
            "--default_rerank",
            str(max(1, rerank_value)),
            "--default_chunk_s",
            str(max(5, args.chunk_len)),
        ]
        if spans_on:
            chunk_cmd.append("--default_spans_on")
        chunk_cmd.extend(
            [
                "--sr",
                str(args.convert_sr),
                "--ffmpeg_bin",
                args.ffmpeg_bin,
            ]
        )
        if desc_text:
            chunk_cmd.extend(["--description", desc_text])
        print("  Running chunked command:", " ".join(chunk_cmd))
        try:
            subprocess.run(chunk_cmd, check=True)
            safe_empty_cuda(device)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"  Chunked processing failed: {exc}")
            safe_empty_cuda(device)
            return False

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matched: {input_dir / args.pattern}")
        return

    # Device + memory cap
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_frac), device=0)
            print(f"CUDA memory fraction cap: {args.gpu_mem_frac}")
        except Exception as e:
            print(f"(warn) Could not set CUDA memory fraction cap: {e}")

    print(f"Model dir: {args.model_dir}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Cache:  {cache_dir}")
    print("-" * 80)

    # Load once (reduces churn)
    model = SAMAudio.from_pretrained(args.model_dir).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.model_dir)

    # Interactive defaults (can adjust at start)
    default_rerank = args.default_rerank
    default_spans = True if args.default_spans_on else False
    default_convert = True if args.auto_convert_16k else False
    default_gpu_frac = args.gpu_mem_frac

    # Optional: let you adjust defaults once at the start of the run
    if prompt_bool("Adjust batch defaults before starting?", False):
        default_gpu_frac = prompt_float("GPU memory fraction cap", default_gpu_frac, 0.10, 1.0)
        if device == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(float(default_gpu_frac), device=0)
                print(f"Set CUDA memory fraction cap: {default_gpu_frac}")
            except Exception as e:
                print(f"(warn) Could not set CUDA memory fraction cap: {e}")
        default_rerank = prompt_int("Default reranking candidates", default_rerank, 1)
        default_spans = prompt_bool("Default predict_spans?", default_spans)
        default_convert = prompt_bool("Default auto-convert to mono 16k?", default_convert)

    last_desc = None

    for idx, in_path in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {in_path.name}")

        if not prompt_bool("Process this file?", True):
            print("  Skipped.")
            continue

        desc = prompt_str(
            "Description (e.g. 'woman speaking, remove cicadas and insects')",
            default=last_desc
        )
        last_desc = desc

        rerank = prompt_int("Reranking candidates (>=1)", default_rerank, 1)
        spans = prompt_bool("predict_spans?", default_spans)
        do_convert = prompt_bool("Convert to mono 16k WAV first?", default_convert)

        duration = audio_duration_seconds(in_path)
        if args.auto_chunk and duration and duration >= args.chunk_threshold_s:
            reason = (
                f"  Duration {duration:.1f}s exceeds threshold "
                f"{args.chunk_threshold_s}s -> switching to chunked runner"
            )
            if not launch_chunk_runner(in_path, desc, rerank, spans, reason):
                print("  Chunked runner failed; skipping file.")
            continue

        proc_path = in_path
        conversion_enabled = do_convert
        converted_path = None

        def ensure_conversion():
            nonlocal proc_path, converted_path
            converted_path = cache_dir / f"{in_path.stem}_mono{args.convert_sr}.wav"
            if args.clear_cache or not converted_path.exists():
                print(f"  Converting -> {converted_path.name}")
                run_ffmpeg_convert(
                    in_path,
                    converted_path,
                    sample_rate=args.convert_sr,
                    channels=1,
                    ffmpeg_bin=args.ffmpeg_bin,
                )
            else:
                print(f"  Using cached conversion -> {converted_path.name}")
            proc_path = converted_path

        if conversion_enabled:
            ensure_conversion()

        result = None
        last_err = None
        chunked_success = False
        batch = None

        try:
            while True:
                batch = processor(audios=[str(proc_path)], descriptions=[desc]).to(device)

                attempt_settings = [
                    (max(1, rerank), spans),
                    (1, spans) if rerank != 1 else None,
                    (1, False) if spans else None,
                ]
                attempt_settings = [x for x in attempt_settings if x is not None]

                for rr, sp in attempt_settings:
                    try:
                        with torch.inference_mode():
                            result = model.separate(
                                batch,
                                predict_spans=sp,
                                reranking_candidates=max(1, rr)
                            )
                        if (rr, sp) != (rerank, spans):
                            print(f"  (fallback) ran with rerank={rr}, predict_spans={sp}")
                        break
                    except RuntimeError as e:
                        last_err = e
                        msg = str(e).lower()
                        if "out of memory" in msg:
                            print("  CUDA OOM; trying safer settings...")
                            safe_empty_cuda(device)
                            continue
                        else:
                            print(f"  RuntimeError: {e}")
                            result = None
                            break

                if result is not None:
                    break

                # All attempts failed
                if not conversion_enabled:
                    if prompt_bool("  OOM persists. Convert to mono 16k and retry?", True):
                        conversion_enabled = True
                        ensure_conversion()
                        continue
                if last_err and "out of memory" in str(last_err).lower():
                    chunk_reason = (
                        "  OOM persists even after safer settings -> chunked runner fallback"
                    )
                    chunked_success = launch_chunk_runner(
                        in_path, desc, rerank, spans, chunk_reason
                    )
                else:
                    print(f"  Skipping {in_path.name}. Last error: {last_err}")
                break

            if chunked_success:
                continue

            if result is None:
                continue

            # Save outputs
            sr = int(processor.audio_sampling_rate)
            out_base = in_path.stem
            out_target = output_dir / f"{out_base}_target.wav"
            out_resid  = output_dir / f"{out_base}_residual.wav"

            sf.write(out_target, tensor_to_numpy(result.target), sr, subtype="PCM_16")
            sf.write(out_resid,  tensor_to_numpy(result.residual), sr, subtype="PCM_16")

            print(f"  Wrote: {out_target.name}, {out_resid.name} @ {sr} Hz")

        finally:
            # Cleanup between files to reduce fragmentation/creep
            if batch is not None:
                del batch
            if result is not None:
                del result
            safe_empty_cuda(device)

    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
