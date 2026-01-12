import argparse
import gc
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

# Reduce CUDA fragmentation before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import soundfile as sf
import torch

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio_utils import env_path, run_ffmpeg_convert, tensor_to_numpy

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DEFAULT_MODEL_DIR = env_path(
    "SAM_MODEL_DIR", Path.home() / "models" / "sam-audio-large-tv"
)
DEFAULT_WORK_DIR = env_path("SAM_WORK_DIR", Path("./audio_work"))
DEFAULT_FFMPEG_BIN = os.environ.get("SAM_FFMPEG_BIN", "ffmpeg")


def prompt_path(prompt: str, default: Path) -> Path:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        raw = raw or str(default)
        path = Path(raw).expanduser()
        if path:
            return path


def prompt_text(prompt: str) -> str:
    while True:
        value = input(f"{prompt}: ").strip()
        if value:
            return value


def find_audio_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def ffmpeg_split_wav(
    in_wav: Path, out_dir: Path, chunk_s: int, ffmpeg_bin: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.wav"):
        old.unlink(missing_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(in_wav),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_s),
        "-reset_timestamps",
        "1",
        str(out_dir / "chunk_%06d.wav"),
    ]
    run_subprocess(cmd)


def concat_wavs_with_ffmpeg(wavs: Iterable[Path], out_path: Path, ffmpeg_bin: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_list = out_path.parent / f".concat_{out_path.stem}.txt"
    lines = []
    for w in wavs:
        quoted = str(w).replace("'", "'\\''")
        lines.append(f"file '{quoted}'")
    temp_list.write_text("\n".join(lines) + "\n")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(temp_list),
        "-c",
        "copy",
        str(out_path),
    ]
    try:
        run_subprocess(cmd)
    finally:
        temp_list.unlink(missing_ok=True)


def run_subprocess(cmd):
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        err = proc.stderr[-2000:] or proc.stdout[-2000:]
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\n{err}")
    return proc


def safe_empty_cuda(device: str):
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


def run_separation(model, processor, path: Path, desc: str, device: str):
    batch = processor(audios=[str(path)], descriptions=[desc]).to(device)
    attempt_settings = [(2, False), (1, False)]
    last_err = None
    for rerank, spans in attempt_settings:
        try:
            with torch.inference_mode():
                return model.separate(
                    batch,
                    predict_spans=spans,
                    reranking_candidates=max(1, rerank),
                )
        except RuntimeError as exc:
            msg = str(exc).lower()
            last_err = exc
            if "out of memory" in msg:
                safe_empty_cuda(device)
                continue
            raise
    raise RuntimeError(f"Failed to separate chunk: {last_err}")


def process_short_file(
    model,
    processor,
    converted_wav: Path,
    desc: str,
    output_dir: Path,
    base_name: str,
    device: str,
):
    result = run_separation(model, processor, converted_wav, desc, device)
    sr = int(processor.audio_sampling_rate)
    target_path = output_dir / f"{base_name}_target.wav"
    residual_path = output_dir / f"{base_name}_residual.wav"
    sf.write(target_path, tensor_to_numpy(result.target), sr, subtype="PCM_16")
    sf.write(residual_path, tensor_to_numpy(result.residual), sr, subtype="PCM_16")
    print(f"  Wrote {target_path.name} / {residual_path.name}")


def process_long_file(
    model,
    processor,
    converted_wav: Path,
    desc: str,
    output_dir: Path,
    base_name: str,
    work_dir: Path,
    device: str,
    ffmpeg_bin: str,
    chunk_len: int,
):
    chunk_dir = work_dir / "chunks"
    out_target = work_dir / "target_chunks"
    out_resid = work_dir / "residual_chunks"
    for d in (chunk_dir, out_target, out_resid):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print(f"  Splitting into {chunk_len}s chunks...")
    ffmpeg_split_wav(converted_wav, chunk_dir, chunk_len, ffmpeg_bin)
    chunk_files = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunk_files:
        print("  No chunks produced; skipping.")
        return

    target_paths = []
    residual_paths = []
    print(f"  Processing {len(chunk_files)} chunk(s)...")
    for idx, chunk in enumerate(chunk_files, start=1):
        print(f"    Chunk {idx}/{len(chunk_files)} -> {chunk.name}")
        try:
            result = run_separation(model, processor, chunk, desc, device)
        except RuntimeError as exc:
            print(f"    Failed chunk {chunk.name}: {exc}")
            return

        sr = int(processor.audio_sampling_rate)
        t_path = out_target / f"{chunk.stem}_target.wav"
        r_path = out_resid / f"{chunk.stem}_residual.wav"
        sf.write(t_path, tensor_to_numpy(result.target), sr, subtype="PCM_16")
        sf.write(r_path, tensor_to_numpy(result.residual), sr, subtype="PCM_16")
        target_paths.append(t_path)
        residual_paths.append(r_path)
        safe_empty_cuda(device)

    final_target = output_dir / f"{base_name}_target.wav"
    final_residual = output_dir / f"{base_name}_residual.wav"
    print("  Concatenating chunks...")
    concat_wavs_with_ffmpeg(target_paths, final_target, ffmpeg_bin)
    concat_wavs_with_ffmpeg(residual_paths, final_residual, ffmpeg_bin)
    print(f"  Wrote {final_target.name} / {final_residual.name}")


def main():
    ap = argparse.ArgumentParser(description="Simple SAM-Audio batch processor")
    ap.add_argument("--input_dir", help="Directory with source audio files")
    ap.add_argument("--output_dir", help="Directory for separated WAVs")
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--work_dir", default=str(DEFAULT_WORK_DIR))
    ap.add_argument("--ffmpeg_bin", default=DEFAULT_FFMPEG_BIN)
    ap.add_argument("--description", help="Text description used for separation")
    ap.add_argument(
        "--gpu_mem_frac",
        type=float,
        default=0.70,
        help="CUDA memory fraction cap (0 disables the cap)",
    )
    ap.add_argument(
        "--chunk_threshold",
        type=int,
        default=30,
        help="Use chunked mode when duration exceeds this many seconds",
    )
    ap.add_argument(
        "--chunk_len",
        type=int,
        default=20,
        help="Chunk length in seconds for long files",
    )
    ap.add_argument(
        "--convert_sr",
        type=int,
        default=16000,
        help="Sample rate for mono conversion",
    )
    args = ap.parse_args()

    input_dir = (
        Path(args.input_dir).expanduser()
        if args.input_dir
        else prompt_path("Input directory", Path("./audio_input"))
    )
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else prompt_path("Output directory", Path("./audio_output"))
    )
    description = (
        args.description if args.description else prompt_text("Describe the target audio")
    )
    model_dir = Path(args.model_dir).expanduser()
    work_root = Path(args.work_dir).expanduser()
    ffmpeg_bin = args.ffmpeg_bin

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No audio files found under {input_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        gpu_mem_frac = float(max(0.0, args.gpu_mem_frac))
        if gpu_mem_frac > 0:
            try:
                torch.cuda.set_per_process_memory_fraction(gpu_mem_frac, device=0)
                print(f"CUDA memory fraction cap: {gpu_mem_frac}")
            except Exception as exc:
                print(f"(warn) Could not set CUDA memory fraction cap: {exc}")
    model = SAMAudio.from_pretrained(model_dir).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(model_dir)

    for idx, src in enumerate(audio_files, start=1):
        print(f"\n[{idx}/{len(audio_files)}] {src.relative_to(input_dir)}")
        file_work = work_root / src.stem
        if file_work.exists():
            shutil.rmtree(file_work)
        file_work.mkdir(parents=True, exist_ok=True)

        converted = file_work / f"{src.stem}_mono{args.convert_sr}.wav"
        print("  Converting to mono 16 kHz...")
        run_ffmpeg_convert(
            src,
            converted,
            sample_rate=args.convert_sr,
            channels=1,
            ffmpeg_bin=ffmpeg_bin,
        )

        info = sf.info(str(converted))
        duration = float(info.duration)
        base_name = src.stem

        if duration > args.chunk_threshold:
            print(
                f"  Duration {duration:.1f}s exceeds threshold "
                f"{args.chunk_threshold}s -> chunked mode"
            )
            try:
                process_long_file(
                    model,
                    processor,
                    converted,
                    description,
                    output_dir,
                    base_name,
                    file_work,
                    device,
                    ffmpeg_bin,
                    args.chunk_len,
                )
            finally:
                safe_empty_cuda(device)
        else:
            try:
                process_short_file(
                    model,
                    processor,
                    converted,
                    description,
                    output_dir,
                    base_name,
                    device,
                )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg:
                    print(
                        "  Encountered CUDA OOM during unchunked run, "
                        f"retrying with {args.chunk_len}s chunks..."
                    )
                    process_long_file(
                        model,
                        processor,
                        converted,
                        description,
                        output_dir,
                        base_name,
                        file_work,
                        device,
                        ffmpeg_bin,
                        args.chunk_len,
                    )
                else:
                    raise
            finally:
                safe_empty_cuda(device)

        shutil.rmtree(file_work, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
