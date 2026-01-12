import os
# Must be set BEFORE importing torch to reduce CUDA fragmentation issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import argparse
import sys
import subprocess
import gc
import re
import shutil

import torch
import soundfile as sf

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio_utils import tensor_to_numpy, run_ffmpeg_convert, env_path

DEFAULT_INPUT_DIR = env_path("SAM_INPUT_DIR", Path("./audio_input"))
DEFAULT_OUTPUT_DIR = env_path("SAM_OUTPUT_DIR", Path("./audio_output"))
DEFAULT_WORK_DIR = env_path("SAM_WORK_DIR", Path("./audio_work"))
DEFAULT_MODEL_DIR = env_path(
    "SAM_MODEL_DIR", Path.home() / "models" / "sam-audio-large-tv"
)
DEFAULT_FFMPEG_BIN = os.environ.get("SAM_FFMPEG_BIN", "ffmpeg")

def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n  {' '.join(cmd)}\n\n{proc.stderr[-3000:]}")
    return proc

def prompt_str(prompt, default=None):
    if default is None:
        s = input(f"{prompt}: ").strip()
        while not s:
            s = input(f"{prompt} (cannot be empty): ").strip()
        return s
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default

def prompt_int(prompt, default, min_value=1, max_value=None):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
            if v < min_value:
                print(f"  Must be >= {min_value}.")
                continue
            if max_value is not None and v > max_value:
                print(f"  Must be <= {max_value}.")
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
                print(f"  Must be between {min_value} and {max_value}.")
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

def ffmpeg_split_wav(in_wav: Path, out_dir: Path, chunk_s: int, ffmpeg_bin: str):
    """
    Split WAV into fixed-length WAV chunks using ffmpeg segment muxer.
    Produces chunk_000000.wav, chunk_000001.wav, ...
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
    run(cmd)

def concat_wavs_with_ffmpeg(wavs, out_path: Path, ffmpeg_bin: str):
    """
    Concatenate WAVs via ffmpeg concat demuxer.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.parent / f".concat_{out_path.stem}.txt"
    # ffmpeg concat demuxer needs "file 'path'"
    lines = []
    for w in wavs:
        # quote single quotes safely
        p = str(w).replace("'", "'\\''")
        lines.append(f"file '{p}'")
    list_file.write_text("\n".join(lines) + "\n")

    cmd = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)]
    run(cmd)
    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass

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

def main():
    ap = argparse.ArgumentParser(description="Interactive batch SAM-Audio separation with chunking")
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--pattern", default="*.wav", help="Glob pattern for input_dir")
    ap.add_argument("--offline", action="store_true", help="Force HF/Transformers offline mode (cache only)")
    ap.add_argument("--gpu_mem_frac", type=float, default=0.70, help="CUDA memory fraction cap (0-1)")
    ap.add_argument("--default_rerank", type=int, default=2, help="Default reranking candidates (>=1)")
    ap.add_argument("--default_spans_on", action="store_true", help="Default predict_spans ON")
    ap.add_argument("--default_chunk_s", type=int, default=20, help="Default chunk length in seconds")
    ap.add_argument("--sr", type=int, default=16000, help="Conversion sample rate (default 16000)")
    ap.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing raw audio")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for concatenated outputs")
    ap.add_argument("--work_dir", default=str(DEFAULT_WORK_DIR), help="Working directory for temp files")
    ap.add_argument("--ffmpeg_bin", default=DEFAULT_FFMPEG_BIN, help="ffmpeg binary to use")
    ap.add_argument("--clean_work_dir", action="store_true", help="Wipe work_dir before processing")
    ap.add_argument("--keep_work_dir", action="store_true", help="Keep per-file work directories (debugging)")
    ap.add_argument("--description", help="Default description to pre-fill prompts")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matched: {input_dir / args.pattern}")
        return

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
    print(f"Work:   {work_dir}")
    print("-" * 90)

    # Load once
    model = SAMAudio.from_pretrained(args.model_dir).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.model_dir)

    default_spans = True if args.default_spans_on else False
    last_desc = args.description

    for idx, in_path in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {in_path.name}")

        if not prompt_bool("Process this file?", True):
            print("  Skipped.")
            continue

        desc = prompt_str("Description", default=last_desc)
        last_desc = desc

        rerank = prompt_int("Reranking candidates (>=1)", default=args.default_rerank, min_value=1, max_value=16)
        spans = prompt_bool("predict_spans?", default=default_spans)
        chunk_s = prompt_int("Chunk length seconds", default=args.default_chunk_s, min_value=5, max_value=300)

        # Prepare working dirs for this file
        stem = in_path.stem
        file_work = work_dir / stem
        if file_work.exists():
            shutil.rmtree(file_work)

        try:
            conv_wav = file_work / f"{stem}_mono{args.sr}.wav"
            chunks_dir = file_work / "chunks"
            out_chunks_target = file_work / "out_chunks_target"
            out_chunks_resid = file_work / "out_chunks_residual"

            for d in (chunks_dir, out_chunks_target, out_chunks_resid):
                d.mkdir(parents=True, exist_ok=True)

            print("  Converting to mono 16k (stable input)...")
            run_ffmpeg_convert(
                in_path,
                conv_wav,
                sample_rate=args.sr,
                channels=1,
                ffmpeg_bin=args.ffmpeg_bin,
            )

            print(f"  Splitting into {chunk_s}s chunks...")
            # Clear any old chunks
            for old in chunks_dir.glob("chunk_*.wav"):
                old.unlink(missing_ok=True)
            ffmpeg_split_wav(
                conv_wav, chunks_dir, chunk_s=chunk_s, ffmpeg_bin=args.ffmpeg_bin
            )

            chunk_files = sorted(chunks_dir.glob("chunk_*.wav"))
            if not chunk_files:
                print("  No chunks produced. Skipping file.")
                continue

            # Process each chunk
            target_chunk_paths = []
            resid_chunk_paths = []

            print(f"  Processing {len(chunk_files)} chunk(s)...")
            for c_i, chunk_path in enumerate(chunk_files, start=1):
                print(f"    Chunk {c_i}/{len(chunk_files)}: {chunk_path.name}")

                batch = None
                result = None
                try:
                    batch = processor(audios=[str(chunk_path)], descriptions=[desc]).to(device)

                    # Try once; on OOM retry with safer settings
                    attempt_settings = [
                        (rerank, spans),
                        (1, False) if (rerank != 1 or spans) else None
                    ]
                    attempt_settings = [x for x in attempt_settings if x is not None]

                    ok = False
                    last_err = None
                    for (rr, sp) in attempt_settings:
                        try:
                            with torch.inference_mode():
                                result = model.separate(batch, predict_spans=sp, reranking_candidates=max(1, rr))
                            ok = True
                            break
                        except RuntimeError as e:
                            last_err = e
                            msg = str(e).lower()
                            if "out of memory" in msg:
                                print("      CUDA OOM; retrying with rerank=1 and predict_spans=False...")
                                safe_empty_cuda(device)
                                continue
                            else:
                                break

                    if not ok:
                        print("      Failed on this chunk. Error:")
                        print(f"      {last_err}")
                        print("      Skipping remaining chunks for this file.")
                        target_chunk_paths = []
                        resid_chunk_paths = []
                        break

                    # Save chunk outputs
                    sr_out = int(processor.audio_sampling_rate)
                    out_t = out_chunks_target / f"{chunk_path.stem}_target.wav"
                    out_r = out_chunks_resid / f"{chunk_path.stem}_residual.wav"
                    sf.write(out_t, tensor_to_numpy(result.target), sr_out, subtype="PCM_16")
                    sf.write(out_r, tensor_to_numpy(result.residual), sr_out, subtype="PCM_16")
                    target_chunk_paths.append(out_t)
                    resid_chunk_paths.append(out_r)

                finally:
                    # Cleanup per chunk - ALWAYS happens even on exceptions
                    if batch is not None:
                        del batch
                    if result is not None:
                        del result
                    safe_empty_cuda(device)

            if not target_chunk_paths:
                print("  No outputs produced for this file (chunk processing failed).")
                continue

            # Concatenate chunk outputs
            final_target = output_dir / f"{stem}_target.wav"
            final_resid = output_dir / f"{stem}_residual.wav"

            print("  Concatenating target chunks...")
            concat_wavs_with_ffmpeg(
                target_chunk_paths, final_target, ffmpeg_bin=args.ffmpeg_bin
            )

            print("  Concatenating residual chunks...")
            concat_wavs_with_ffmpeg(
                resid_chunk_paths, final_resid, ffmpeg_bin=args.ffmpeg_bin
            )

            print(f"  Wrote final:\n    {final_target.name}\n    {final_resid.name}")

            # final cleanup
            safe_empty_cuda(device)

        finally:
            if not args.keep_work_dir:
                shutil.rmtree(file_work, ignore_errors=True)

    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
