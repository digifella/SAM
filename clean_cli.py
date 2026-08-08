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
import re
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

from sam_audio_utils.denoise import denoise_file  # noqa: E402
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


# Positive evidence that a SECOND source should be separated out. Anything else
# denoises, because that is the reversible failure: an under-cleaned file is
# still listenable, whereas SAM on single-speaker noise returns an empty stem.
# Only sounds that are unambiguously a SEPARATE source belong here. Words that
# equally often name the RECORDING MEDIUM -- radio, phone, tv, television --
# were removed: "clean up this phone call recording" and "improve this radio
# interview" are cleanup requests, and routing them to SAM sends single-speaker
# noisy audio down the path that returns an empty stem. They are still honoured
# via _MEDIUM_WORDS below, but only alongside evidence the device is audibly
# playing -- dropping them outright lost "a tv playing in the background".
_MIXTURE_WORDS = (
    "guitar", "piano", "drum", "bass", "violin", "music", "song", "instrument",
    "dog", "bark", "bird", "engine", "traffic", "siren", "alarm", "applause",
    "crowd", "typing", "keyboard",
)
_MIXTURE_PHRASES = (" over ", " behind ", " through ", " on top of ", " against ")

# Matched on WORD BOUNDARIES, not as bare substrings. Substring matching read
# "engine" out of "engineer", "bass" out of "embassy" and "music" out of
# "musician", so "an interview with an engineer" was routed to SAM -- the same
# misroute as the recording-medium words above, and in the direction that costs
# the most: a false "separate" returns a near-empty stem, whereas a false
# "denoise" merely under-cleans a file that is still listenable.
# A trailing "s" is allowed so plurals ("two guitars") still count. Other
# inflections deliberately do NOT match: allowing "-ed"/"-ing" would read
# "alarm" out of "alarming" and "crowd" out of "crowded", reintroducing the bug.
# The cost is that a bare "barking outside" now denoises; that is the safe
# direction, and an explicit method="separate" overrides it.
_MIXTURE_RE = re.compile(r"\b(" + "|".join(_MIXTURE_WORDS) + r")s?\b")

# A device that is EITHER the recording medium or an audible second source.
# On its own it means nothing ("this phone call recording"); paired with a cue
# that it is actually playing, it is a genuine mixture.
_MEDIUM_WORDS = ("tv", "television", "radio", "phone", "stereo")
_MEDIUM_RE = re.compile(r"\b(" + "|".join(_MEDIUM_WORDS) + r")s?\b")
_BACKGROUND_CUES = ("background", "playing", "blaring", "is on", "next room",
                    "in another room")

# SAM extracts the sound you DESCRIBE -- it does not follow instructions. So
# "remove the barking dog" makes it return the DOG as target.wav and push the
# voice into residual.wav, then report success, with the speech-band guard
# disarmed because the description never mentions speech. That is this project's
# founding failure reachable through ordinary English, so removal phrasing routes
# to denoise: it under-cleans, which is the reversible direction.
# To extract a source, NAME it ("the guitar") rather than asking to remove it.
_REMOVAL_CUES = (
    "remove", "delete", "strip", "suppress", "eliminate", "mute", "silence",
    "take out", "cut out", "get rid of", "filter out", "without the",
    "minus the", "reduce", "kill", "drown out", "clean out", "take away",
    "less of",
    "strip out",  # Must be listed explicitly for exact-phrase extraction; "strip" alone misses two-word variant. Routing is neutral: "strip out" ⊃ "strip" so choose_method's membership test is unchanged.
)

# Leading determiners left behind once the cue is gone: "remove the guitar"
# strips to "the guitar", and SAM does better with the bare source name.
_LEADING_ARTICLE_RE = re.compile(
    r"^(?:the|a|an|any|all|that|this|those|these)\s+", re.IGNORECASE)


def strip_removal_cue(description: str) -> str:
    """Derive SAM's extraction prompt from a removal request.

    SAM extracts the sound it is DESCRIBED, so the cue must not survive:
    "remove the barking dog" has to reach SAM as "barking dog". The remainder
    is not re-flowed as English -- a cue in the middle of a sentence leaves a
    clumsy prompt, which is accepted so long as the named source survives.

    Returns "" when nothing usable is left; the caller vetoes the route.
    """
    text = (description or "").strip()
    if not text:
        return ""
    low = text.lower()
    # Longest cue first: "strip out" must win over the shorter "strip" cue it contains.
    for cue in sorted(_REMOVAL_CUES, key=len, reverse=True):
        i = low.find(cue)
        if i == -1:
            continue
        text = f"{text[:i]} {text[i + len(cue):]}".strip()
        break
    text = _LEADING_ARTICLE_RE.sub("", text.strip() + " ", count=1).rstrip()
    return " ".join(text.split())


def choose_method(description: str, explicit: str = "auto") -> str:
    """Pick the processing method. Explicit always beats inference."""
    if explicit in ("denoise", "separate", "remove"):
        return explicit
    if explicit != "auto":
        raise ValueError(
            f"unknown method {explicit!r}; expected auto, denoise, separate or remove")
    text = (description or "").lower()
    # Checked FIRST: a removal request names the sound to DISCARD. If that sound
    # is one SAM can extract, run SAM and hand back the residual; if it is only
    # "noise" or "hiss", there is no source to extract, so denoise instead.
    if any(r in text for r in _REMOVAL_CUES):
        if _MIXTURE_RE.search(text):
            return "remove"
        if _MEDIUM_RE.search(text) and any(c in text for c in _BACKGROUND_CUES):
            return "remove"
        return "denoise"
    if _MIXTURE_RE.search(text):
        return "separate"
    if any(p in f" {text} " for p in _MIXTURE_PHRASES):
        return "separate"
    if _MEDIUM_RE.search(text) and any(c in text for c in _BACKGROUND_CUES):
        return "separate"
    return "denoise"


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
    # out_dir must exist before anything below can write status.json into it —
    # if this itself fails there is nowhere to record an error, so let it raise.
    out_dir.mkdir(parents=True, exist_ok=True)

    cancel = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())

    write_status(out_dir, "running", pid=os.getpid(), started_at=_now(),
                 input=input_path.name)
    try:
        # job-json parsing lives inside the try so a malformed/partial file
        # still lands as state:"error" instead of leaving status.json stuck
        # on "running" forever (the bridge treats a missing/never-updated
        # status.json as still-pending).
        payload = json.loads(Path(args.job_json).read_text())
        method = choose_method(payload.get("description", ""),
                               payload.get("method", "auto"))
        strength = payload.get("strength", "normal")
        emit(5, "route", f"method={method}")

        if method == "denoise":
            # No GPU lock, no model: this path is pure CPU DSP, so it runs even
            # while a SAM job holds the card.
            work = out_dir / "denoise_work"
            work.mkdir(parents=True, exist_ok=True)
            target_wav = work / "target.wav"
            residual_wav = work / "residual.wav"
            meta = denoise_file(input_path, target_wav, residual_wav,
                                strength=strength,
                                progress_cb=lambda p, m: emit(p, "denoise", m))
            meta["description"] = payload.get("description", "")
            meta["input_filename"] = input_path.name
            zip_src = work / "result.zip"
            with zipfile.ZipFile(zip_src, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(target_wav, "target.wav")
                zf.write(residual_wav, "residual.wav")
                zf.writestr("metadata.json", json.dumps(meta, indent=2))
            result = {"output_data": meta, "output_file": zip_src}
        else:
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
        opus_error: str | None = None
        if args.opus:
            emit(95, "encode", "encoding target.wav to Opus")
            try:
                with tempfile.TemporaryDirectory() as td:
                    with zipfile.ZipFile(final_zip) as zf:
                        zf.extract("target.wav", td)
                    ogg = encode_opus(Path(td) / "target.wav", out_dir / "target.ogg")
            except Exception as exc:
                # Separation already succeeded and result.zip is valid on disk —
                # a broken/missing ffmpeg is a delivery problem, not a job
                # failure. Degrade like post_discord_webhook's own network
                # failures do: keep state:"done", surface a warning, notify
                # without the audio attachment.
                opus_error = str(exc)
                ogg = None
                emit(95, "encode", f"opus encode failed, continuing without it: {opus_error}")

        meta = result.get("output_data", {}) or {}
        if opus_error:
            meta = dict(meta)
            prior_warning = meta.get("warning")
            meta["warning"] = (f"{prior_warning}; " if prior_warning else "") + f"opus encode failed: {opus_error}"
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
        # The success path removes this; on the error path it was being left
        # behind, and a failure after the first sf.write strands a full-size WAV.
        shutil.rmtree(out_dir / "denoise_work", ignore_errors=True)
        write_status(out_dir, "error", error=str(exc), finished_at=_now())
        _notify(args, f"❌ Audio cleanup of `{input_path.name}` failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
