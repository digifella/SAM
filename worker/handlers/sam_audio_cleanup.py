from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
import torch
from sam_audio import SAMAudioProcessor

from sam_audio_local.loader import load_sam_audio_optimized

from run_sam_interactive import (
    apply_cuda_memory_fraction_safely,
    aggressive_cleanup,
    count_chunks,
    get_audio_duration,
    patch_sam_audio_model,
    process_audio_file,
)

logger = logging.getLogger(__name__)


class _ModelCache:
    _lock = threading.Lock()
    _model = None
    _processor = None
    _model_dir = ""
    _device = ""

    @classmethod
    def get(
        cls,
        model_dir: Path,
        device_pref: str,
        memory_fraction: float,
    ):
        with cls._lock:
            device = _resolve_device(device_pref)
            model_dir_str = str(model_dir.resolve())

            if cls._model is None or cls._processor is None or cls._model_dir != model_dir_str or cls._device != device:
                logger.info("Loading SAM-Audio model: dir=%s device=%s", model_dir_str, device)
                if device == "cuda":
                    effective_fraction = apply_cuda_memory_fraction_safely(
                        memory_fraction,
                        logger=logger,
                        context="_ModelCache.get",
                    )
                    logger.info(
                        "CUDA memory fraction requested=%.3f effective=%.3f",
                        float(memory_fraction),
                        float(effective_fraction),
                    )

                model = load_sam_audio_optimized(model_dir_str, device)
                processor = SAMAudioProcessor.from_pretrained(model_dir_str)
                patch_sam_audio_model(model)

                cls._model = model
                cls._processor = processor
                cls._model_dir = model_dir_str
                cls._device = device

            return cls._model, cls._processor, cls._device

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._model = None
            cls._processor = None
            cls._model_dir = ""
            cls._device = ""
        aggressive_cleanup()


def _resolve_device(device_pref: str) -> str:
    p = str(device_pref or "auto").strip().lower()
    if p == "cpu":
        return "cpu"
    if p == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    return max(min_value, min(max_value, f))


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        i = int(value)
    except Exception:
        return default
    return max(min_value, min(max_value, i))


def _apply_trial_cut(input_path: Path, trial_seconds: int, ffmpeg_bin: str, work_dir: Path) -> Path:
    trimmed = work_dir / f"trial_{trial_seconds}s.wav"
    cmd = [
        ffmpeg_bin,
        "-i", str(input_path),
        "-t", str(trial_seconds),
        "-y",
        str(trimmed),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg trial cut failed: {proc.stderr[-500:]}")
    return trimmed


def _apply_peak_normalize(path: Path, percent: float) -> None:
    target_peak = max(0.0, min(1.0, percent / 100.0))
    if target_peak <= 0.0:
        return

    data, sr = sf.read(path, always_2d=False)
    arr = np.asarray(data)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 0.0:
        return

    gain = target_peak / peak
    out = np.clip(arr * gain, -1.0, 1.0)
    sf.write(path, out, sr, subtype="PCM_16")


LOUDNORM_I = -16.0
LOUDNORM_TP = -3.0
LOUDNORM_LRA = 11.0
_LOUDNORM_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def _parse_loudnorm_json(stderr_text: str) -> dict:
    """Extract the measurement JSON that ffmpeg loudnorm prints to stderr."""
    start = stderr_text.rfind("{")
    end = stderr_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no loudnorm JSON in ffmpeg output: {stderr_text[-300:]}")
    try:
        measured = json.loads(stderr_text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"unparseable loudnorm JSON: {exc}") from exc
    missing = [k for k in _LOUDNORM_KEYS if k not in measured]
    if missing:
        raise ValueError(f"loudnorm JSON missing keys: {missing}")
    return measured


def _transcode_audio(path: Path, sample_rate: Optional[int], channels: Optional[int], ffmpeg_bin: str) -> Path:
    if sample_rate is None and channels is None:
        return path

    tmp = path.with_name(f"{path.stem}_transcoded.wav")
    cmd = [ffmpeg_bin, "-i", str(path)]
    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])
    if channels is not None:
        cmd.extend(["-ac", str(channels)])
    cmd.extend(["-y", str(tmp)])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {proc.stderr[-500:]}")

    tmp.replace(path)
    return path


def _find_single_output(output_dir: Path, suffix: str) -> Path:
    candidates = sorted(
        [p for p in output_dir.glob(f"*{suffix}.wav") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"Missing SAM output file ending with {suffix}.wav")
    return candidates[0]


def _safe_progress(progress_cb: Optional[Callable[[float, str, Optional[str]], None]], pct: float, msg: str, stage: str) -> None:
    if progress_cb:
        progress_cb(pct, msg, stage)


def _is_cuda_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text or "cudnn" in text)


def _is_cap_limited_cuda_oom(exc: Exception) -> bool:
    """
    Detect OOM where PyTorch per-process cap is the bottleneck:
    - "... Tried to allocate X MiB ..."
    - "... Y GiB allowed; Of the allocated memory Z GiB is allocated ..."
    """
    text = str(exc)
    if not _is_cuda_oom(exc):
        return False

    alloc_match = re.search(r"Tried to allocate\s+([0-9]+(?:\.[0-9]+)?)\s+MiB", text)
    allowed_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+GiB allowed", text)
    used_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+GiB is allocated by PyTorch", text)
    if not alloc_match or not allowed_match or not used_match:
        return False

    requested_gib = float(alloc_match.group(1)) / 1024.0
    allowed_gib = float(allowed_match.group(1))
    allocated_gib = float(used_match.group(1))
    cap_headroom_gib = max(0.0, allowed_gib - allocated_gib)

    # Leave a small safety buffer; if this fails, retries should lift the cap.
    return cap_headroom_gib + 0.10 < requested_gib


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    if not input_path or not input_path.exists():
        raise ValueError("sam_audio_cleanup requires an input audio file")

    payload = input_data or {}
    description = str(payload.get("description") or "speech").strip() or "speech"
    chunk_duration = _clamp_float(payload.get("chunk_duration"), 60.0, 5.0, 600.0)
    overlap = _clamp_float(payload.get("overlap"), 2.0, 0.0, 30.0)
    if overlap >= chunk_duration:
        overlap = max(0.0, chunk_duration * 0.2)

    convert_to_mono = _as_bool(payload.get("convert_to_mono"), True)
    rerank = _clamp_int(payload.get("rerank"), 1, 1, 8)
    predict_spans = _as_bool(payload.get("predict_spans"), False)

    trial_seconds = _clamp_int(payload.get("trial_seconds"), 0, 0, 86400)
    normalize_percent = _clamp_float(payload.get("normalize_percent"), 0.0, 0.0, 100.0)

    output_sample_rate = payload.get("output_sample_rate")
    if output_sample_rate is not None:
        output_sample_rate = _clamp_int(output_sample_rate, 32000, 8000, 96000)

    output_channels = payload.get("output_channels")
    if output_channels is not None:
        output_channels = _clamp_int(output_channels, 1, 1, 2)

    memory_fraction = _clamp_float(payload.get("memory_fraction"), 0.85, 0.1, 0.98)
    allow_cpu_fallback = _as_bool(payload.get("allow_cpu_fallback"), True)
    ffmpeg_bin = str(payload.get("ffmpeg_bin") or os.environ.get("FFMPEG_BIN") or "ffmpeg")

    model_dir_str = str(payload.get("model_dir") or os.environ.get("SAM_MODEL_DIR") or (Path.home() / "models" / "sam-audio-large-tv"))
    model_dir = Path(model_dir_str)
    if not model_dir.exists():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")

    device_pref = str(payload.get("device") or "auto")

    _safe_progress(progress_cb, 5, "Preparing audio job", "prepare")

    work_dir = Path(tempfile.mkdtemp(prefix=f"sam_audio_job_{job.get('id', 'x')}_"))
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    process_input = input_path
    try:
        if trial_seconds > 0:
            _safe_progress(progress_cb, 10, f"Applying trial cut ({trial_seconds}s)", "preprocess")
            process_input = _apply_trial_cut(input_path, trial_seconds, ffmpeg_bin, work_dir)

        if is_cancelled_cb and is_cancelled_cb():
            raise RuntimeError("Cancelled before processing started")

        _safe_progress(progress_cb, 20, "Loading SAM-Audio model", "model")
        model, processor, device = _ModelCache.get(model_dir, device_pref, memory_fraction)

        duration_seconds = float(get_audio_duration(process_input))
        chunks_estimated = max(1, count_chunks(process_input, chunk_duration, overlap))

        # Auto-retry profile for CUDA OOM:
        # 1) requested settings
        # 2) safer (smaller chunks, rerank off, spans off, lower mem cap)
        # 3) ultra-safe chunking
        attempt_profiles = [
            {
                "chunk_duration": chunk_duration,
                "overlap": overlap,
                "rerank": rerank,
                "predict_spans": predict_spans,
                "memory_fraction": memory_fraction,
                "convert_to_mono": convert_to_mono,
                "label": "requested",
            }
        ]
        if device == "cuda":
            attempt_profiles.append(
                {
                    "chunk_duration": min(chunk_duration, 20.0),
                    "overlap": min(overlap, 1.0),
                    "rerank": 1,
                    "predict_spans": False,
                    "memory_fraction": min(memory_fraction, 0.70),
                    "convert_to_mono": True,
                    "label": "safe-oom-retry",
                }
            )
            attempt_profiles.append(
                {
                    "chunk_duration": min(chunk_duration, 12.0),
                    "overlap": min(overlap, 0.5),
                    "rerank": 1,
                    "predict_spans": False,
                    "memory_fraction": min(memory_fraction, 0.60),
                    "convert_to_mono": True,
                    "label": "ultra-safe-oom-retry",
                }
            )

        last_error: Optional[Exception] = None
        used_profile = attempt_profiles[0]
        ok = False
        cap_relief_memory_fraction = 0.0
        cap_relief_retry_appended = False
        for i, prof in enumerate(attempt_profiles, start=1):
            attempt_memory_fraction = max(float(prof["memory_fraction"]), cap_relief_memory_fraction)
            used_profile = {**prof, "memory_fraction": attempt_memory_fraction}
            _safe_progress(
                progress_cb,
                35,
                f"Running SAM-Audio separation (attempt {i}/{len(attempt_profiles)}: {prof['label']})",
                "separate",
            )
            try:
                ok = process_audio_file(
                    audio_path=process_input,
                    description=description,
                    output_dir=output_dir,
                    model=model,
                    processor=processor,
                    device=device,
                    memory_fraction=attempt_memory_fraction,
                    rerank=int(prof["rerank"]),
                    predict_spans=bool(prof["predict_spans"]),
                    chunk_duration=float(prof["chunk_duration"]),
                    overlap=float(prof["overlap"]),
                    convert_to_mono=bool(prof["convert_to_mono"]),
                    progress_cb=lambda pct, msg: _safe_progress(progress_cb, pct, msg, "separate"),
                    is_cancelled_cb=is_cancelled_cb,
                    raise_on_error=True,
                )
                if ok:
                    break
            except Exception as e:
                last_error = e
                if is_cancelled_cb and is_cancelled_cb():
                    raise
                if not _is_cuda_oom(e):
                    raise
                if device == "cuda" and _is_cap_limited_cuda_oom(e):
                    cap_relief_memory_fraction = max(
                        cap_relief_memory_fraction,
                        min(0.95, max(memory_fraction, 0.90)),
                    )
                    _safe_progress(
                        progress_cb,
                        30,
                        f"CUDA cap-limited OOM detected; raising cap to {cap_relief_memory_fraction:.2f} for retries",
                        "separate",
                    )
                    if i >= len(attempt_profiles) and not cap_relief_retry_appended:
                        attempt_profiles.append(
                            {
                                "chunk_duration": float(prof["chunk_duration"]),
                                "overlap": float(prof["overlap"]),
                                "rerank": int(prof["rerank"]),
                                "predict_spans": bool(prof["predict_spans"]),
                                "memory_fraction": float(cap_relief_memory_fraction),
                                "convert_to_mono": bool(prof["convert_to_mono"]),
                                "label": "cap-relief-retry",
                            }
                        )
                        cap_relief_retry_appended = True
                        _safe_progress(
                            progress_cb,
                            31,
                            "Added one CUDA cap-relief retry before CPU fallback",
                            "separate",
                        )
                if i >= len(attempt_profiles):
                    # Don't raise here; allow post-loop CPU fallback logic to run.
                    break
                _safe_progress(
                    progress_cb,
                    30,
                    f"CUDA OOM detected; retrying with safer settings ({i + 1}/{len(attempt_profiles)})",
                    "separate",
                )
                aggressive_cleanup()

        if not ok:
            if last_error and _is_cuda_oom(last_error) and device == "cuda" and allow_cpu_fallback:
                _safe_progress(progress_cb, 28, "CUDA OOM persists; switching to CPU fallback", "separate")
                _ModelCache.clear()
                model, processor, device = _ModelCache.get(model_dir, "cpu", memory_fraction)
                ok = process_audio_file(
                    audio_path=process_input,
                    description=description,
                    output_dir=output_dir,
                    model=model,
                    processor=processor,
                    device=device,
                    memory_fraction=memory_fraction,
                    rerank=1,
                    predict_spans=False,
                    chunk_duration=min(chunk_duration, 12.0),
                    overlap=min(overlap, 0.5),
                    convert_to_mono=True,
                    progress_cb=lambda pct, msg: _safe_progress(progress_cb, pct, msg, "separate"),
                    is_cancelled_cb=is_cancelled_cb,
                    raise_on_error=True,
                )
                used_profile = {
                    "chunk_duration": min(chunk_duration, 12.0),
                    "overlap": min(overlap, 0.5),
                    "rerank": 1,
                    "predict_spans": False,
                    "memory_fraction": max(memory_fraction, cap_relief_memory_fraction),
                    "convert_to_mono": True,
                    "label": "cpu-fallback",
                }
            if not ok:
                if last_error:
                    raise last_error
                raise RuntimeError("SAM-Audio processing failed")
        if is_cancelled_cb and is_cancelled_cb():
            raise RuntimeError("Cancelled after separation")

        target_path = _find_single_output(output_dir, "_target")
        residual_path = _find_single_output(output_dir, "_residual")

        if normalize_percent > 0:
            _safe_progress(progress_cb, 75, f"Normalizing to {normalize_percent:.1f}% peak", "postprocess")
            _apply_peak_normalize(target_path, normalize_percent)
            _apply_peak_normalize(residual_path, normalize_percent)
            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Cancelled during post-processing")

        if output_sample_rate is not None or output_channels is not None:
            _safe_progress(progress_cb, 82, "Transcoding output format", "postprocess")
            _transcode_audio(target_path, output_sample_rate, output_channels, ffmpeg_bin)
            _transcode_audio(residual_path, output_sample_rate, output_channels, ffmpeg_bin)
            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Cancelled during transcoding")

        info = sf.info(target_path)

        metadata = {
            "job_id": int(job.get("id", 0) or 0),
            "input_filename": str(job.get("input_filename") or input_path.name),
            "description": description,
            "duration_seconds": round(duration_seconds, 3),
            "chunks_processed": chunks_estimated,
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "options_applied": {
                "trial_seconds": trial_seconds,
                "convert_to_mono": convert_to_mono,
                "chunk_duration": float(used_profile["chunk_duration"]),
                "overlap": float(used_profile["overlap"]),
                "rerank": int(used_profile["rerank"]),
                "predict_spans": bool(used_profile["predict_spans"]),
                "normalize_percent": normalize_percent,
                "output_sample_rate": output_sample_rate,
                "output_channels": output_channels,
                "device": device,
                "memory_fraction": float(used_profile["memory_fraction"]),
                "auto_profile": str(used_profile["label"]),
                "allow_cpu_fallback": allow_cpu_fallback,
            },
        }

        zip_path = work_dir / "audio_cleanup_result.zip"
        _safe_progress(progress_cb, 90, "Packaging result ZIP", "package")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(target_path, arcname="target.wav")
            zf.write(residual_path, arcname="residual.wav")
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        _safe_progress(progress_cb, 98, "Result ready for upload", "complete")

        return {
            "output_data": metadata,
            "output_file": zip_path,
        }
    finally:
        aggressive_cleanup()
        for path in output_dir.glob("*.wav") if output_dir.exists() else []:
            path.unlink(missing_ok=True)
        if process_input != input_path:
            process_input.unlink(missing_ok=True)
        archive_path = work_dir / "audio_cleanup_result.zip"
        if archive_path.exists():
            # Keep ZIP until worker uploads it.
            pass
        # temp directory cleanup is handled by worker.py after upload
