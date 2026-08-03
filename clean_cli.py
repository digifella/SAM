#!/usr/bin/env python3
"""CLI wrapper around worker.handlers.sam_audio_cleanup.handle.

Used by the Cortex Suite page (foreground subprocess) and the kb-query-server
bridge (detached spawn for Discord jobs). Progress is emitted as JSON lines on
stdout; final state lands in <out-dir>/status.json.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.handlers.sam_audio_cleanup import handle  # noqa: E402

GPU_LOCK_PATH = Path.home() / ".sam_audio_gpu.lock"
OPUS_BITRATES = ("48k", "24k")  # retry ladder to stay under Discord's 25MB
DISCORD_SIZE_CAP = 24 * 1024 * 1024


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def emit(pct: int, stage: str, message: str, stream=None) -> None:
    line = {"type": "progress", "pct": int(pct), "stage": str(stage),
            "message": str(message), "ts": time.strftime("%H:%M:%S")}
    out = stream or sys.stdout
    out.write(json.dumps(line) + "\n")
    out.flush()


def write_status(out_dir: Path, state: str, **extra) -> None:
    payload = {"state": state, **extra}
    tmp = out_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_dir / "status.json")


@contextmanager
def gpu_lock(timeout_s: int):
    fh = open(GPU_LOCK_PATH, "w")
    t0 = time.monotonic()
    warned = 0.0
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            waited = time.monotonic() - t0
            if waited > timeout_s:
                fh.close()
                raise RuntimeError(f"GPU lock timeout after {int(waited)}s")
            if waited - warned >= 30:
                emit(0, "queue", f"waiting for GPU lock ({int(waited)}s)")
                warned = waited
            time.sleep(2)
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def encode_opus(target_wav: Path, out_ogg: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    for bitrate in OPUS_BITRATES:
        cmd = [ffmpeg_bin, "-y", "-i", str(target_wav), "-ac", "1",
               "-c:a", "libopus", "-b:a", bitrate, str(out_ogg)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"opus encode failed: {proc.stderr[-400:]}")
        if out_ogg.stat().st_size <= DISCORD_SIZE_CAP:
            return out_ogg
    return out_ogg  # oversized even at floor bitrate; webhook will report failure


def post_discord_webhook(url: str, content: str, file_path: Path | None = None,
                         timeout: int = 120) -> bool:
    import requests
    try:
        if file_path is not None:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    url, data={"content": content[:1900]},
                    files={"file": (file_path.name, f, "audio/ogg")}, timeout=timeout)
        else:
            resp = requests.post(url, json={"content": content[:1900]}, timeout=timeout)
        return 200 <= resp.status_code < 300
    except Exception as exc:  # network failure must never fail the job
        emit(99, "notify", f"discord webhook post failed: {exc}")
        return False


def _notify(args, content: str, file_path: Path | None = None) -> None:
    if not args.notify_discord:
        return
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        emit(99, "notify", "DISCORD_WEBHOOK_URL not set; skipping Discord notify")
        return
    ok = post_discord_webhook(url, content, file_path)
    emit(99, "notify", "posted to Discord" if ok else "Discord post FAILED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SAM-Audio cleanup CLI")
    ap.add_argument("--input", required=True)
    ap.add_argument("--job-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--opus", action="store_true")
    ap.add_argument("--notify-discord", action="store_true")
    ap.add_argument("--gpu-lock-timeout", type=int, default=7200)
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.job_json).read_text())

    cancel = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())

    write_status(out_dir, "running", pid=os.getpid(), started_at=_now(),
                 input=input_path.name)
    try:
        with gpu_lock(args.gpu_lock_timeout):
            result = handle(
                input_path=input_path,
                input_data=payload,
                job={"id": 0, "input_filename": input_path.name},
                progress_cb=lambda pct, msg, stage=None: emit(pct, stage or "processing", msg),
                is_cancelled_cb=cancel.is_set,
                work_dir=out_dir,
            )

        zip_src = Path(result["output_file"])
        final_zip = out_dir / "result.zip"
        if zip_src != final_zip:
            shutil.move(str(zip_src), str(final_zip))
        # handle() nests its products in <out_dir>/sam_handler_<id>/ — remove that,
        # but never out_dir itself (status.json lives there).
        if zip_src.parent != out_dir:
            shutil.rmtree(zip_src.parent, ignore_errors=True)

        ogg: Path | None = None
        if args.opus:
            emit(95, "encode", "encoding target.wav to Opus")
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(final_zip) as zf:
                    zf.extract("target.wav", td)
                ogg = encode_opus(Path(td) / "target.wav", out_dir / "target.ogg")

        meta = result.get("output_data", {}) or {}
        summary = (f"✅ Cleaned `{input_path.name}` — "
                   f"{meta.get('duration_seconds', '?')}s of audio, "
                   f"description: \"{meta.get('description', '')}\"")
        if meta.get("warning"):
            summary += f"\n⚠️ {meta['warning']}"
        _notify(args, summary, ogg)

        write_status(out_dir, "done", metadata=meta, finished_at=_now())
        emit(100, "complete", "done")
        return 0
    except Exception as exc:
        write_status(out_dir, "error", error=str(exc), finished_at=_now())
        _notify(args, f"❌ Audio cleanup of `{input_path.name}` failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
