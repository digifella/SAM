# SAM-Audio Quality + Cortex + Hermes Discord Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden sam-audio (bug fixes + verified 45-min capacity), expose it as a Cortex Suite page, and wire a Discord→Hermes(sp4)→fastfella→Discord audio-cleanup loop.

**Architecture:** A shared `clean_cli.py` in sam-audio's venv wraps the existing `sam_audio_cleanup.handle` pipeline; the Cortex page spawns it as a subprocess (JSON-line progress), and the token-gated `kb-query-server.py` bridge spawns it detached for Discord jobs, which push an Opus file back to a channel webhook on completion. A flock serializes GPU access.

**Tech Stack:** Python 3.11, torch 2.6.0+cu124 (fp16, DO NOT upgrade), Streamlit 1.19 (sam venv) / 1.36 (cortex venv), ffmpeg 6.1.1 with libopus, requests 2.32.5, plain pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-08-01-quality-cortex-hermes-integration-design.md`

## Global Constraints

- **Interruption safety:** every task ends in a commit (or an explicitly-marked host-script backup). A reboot between tasks loses nothing. Resume = read this plan's checkboxes + `git log`.
- **GPU:** Quadro RTX 8000, fp16 only (NO bf16). WSL2 real RAM ~23GB — never buffer whole large files in RAM when streaming to disk is possible.
- **Do not upgrade torch (2.6.0+cu124) or streamlit in the sam venv** — the fp16 build works; currency findings are recorded, not chased.
- Paths: sam-audio repo `/home/longboardfella/sam-audio` (venv `.venv/bin/python`), cortex repo `/home/longboardfella/cortex_suite` (venv `venv/bin/python`), bridge `/home/longboardfella/kb-query-server.py` (NOT in git — make `.bak-YYYYMMDD` copies before edits), sp4 via `ssh sp4 -o RemoteCommand=none` (user `paul`).
- Model dir: env `SAM_MODEL_DIR`, default `~/models/sam-audio-large-tv`. Bridge bind `100.118.92.17:7333`, token `Authorization: Bearer $(cat ~/.kb-query-token)`.
- **Subagent policy (per Paul):** implementation/research subagents run on `model: "sonnet"` (or haiku for log reduction). Fable does planning, triage, and final review only.
- Scratchpad for throwaway artifacts: `/tmp/claude-1000/-home-longboardfella-sam-audio/456cf002-c7df-49ad-9c48-620a8286549b/scratchpad` (call it `$SCRATCH`; cleared on reboot — everything needed to resume lives in git).
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Tests run from sam-audio root: `.venv/bin/python -m pytest tests/ -v` (no pytest.ini; unittest-style classes are the house style).

---

## Phase 0 — Review fixes

### Task 1: Fix handler work-dir leak

`handle()` creates `tempfile.mkdtemp` and nothing ever removes it — every job leaks a work dir (with result ZIP) into /tmp on BOTH the worker and Streamlit paths (the closing comment in `handle` blaming worker.py is wrong; worker.py only removes its own dir).

**Files:**
- Modify: `worker/handlers/sam_audio_cleanup.py:318-364` (signature + work_dir creation), `:620-630` (stale comment)
- Modify: `worker/worker.py:236-240` (pass work_dir)
- Modify: `streamlit_app.py:111-117` (pass work_dir)
- Test: `tests/test_handler_workdir.py` (create)

**Interfaces:**
- Produces: `handle(input_path, input_data, job, progress_cb=None, is_cancelled_cb=None, work_dir: Optional[Path] = None)` — when `work_dir` is given, ALL temp products (including the result ZIP) land under `<work_dir>/sam_handler_<jobid>/` and the caller owns cleanup. When None (legacy), mkdtemp as before and the caller must remove `output_file`'s parent tree. Tasks 6 and 10 rely on the `work_dir` parameter.

- [x] **Step 1: Write the failing test**

Create `tests/test_handler_workdir.py`:

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import worker.handlers.sam_audio_cleanup as h


def _fake_process(audio_path, description, output_dir, **kw):
    sr = 16000
    noise = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
    sf.write(Path(output_dir) / "in_target.wav", noise, sr)
    sf.write(Path(output_dir) / "in_residual.wav", noise, sr)
    return True


class HandlerWorkDirTests(unittest.TestCase):
    def test_handle_uses_caller_work_dir_and_never_mkdtemps(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            input_wav = td / "in.wav"
            sf.write(input_wav, np.zeros(16000, dtype=np.float32), 16000)

            with patch.object(h._ModelCache, "get", return_value=(None, None, "cpu")), \
                 patch.object(h, "process_audio_file", side_effect=_fake_process), \
                 patch.object(h, "auto_input_gain",
                              side_effect=lambda p, f, out_dir=None: (p, 0.0)), \
                 patch.object(h, "get_audio_duration", return_value=1.0), \
                 patch.object(h, "count_chunks", return_value=1), \
                 patch.object(h.tempfile, "mkdtemp",
                              side_effect=AssertionError("mkdtemp must not be called")):
                result = h.handle(
                    input_path=input_wav,
                    input_data={"description": "speech"},
                    job={"id": 1},
                    work_dir=td,
                )

            out_zip = Path(result["output_file"])
            self.assertTrue(out_zip.exists())
            self.assertTrue(str(out_zip).startswith(str(td)),
                            f"{out_zip} not under caller work_dir {td}")


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_handler_workdir.py -v`
Expected: FAIL — `handle() got an unexpected keyword argument 'work_dir'`

- [x] **Step 3: Implement**

In `worker/handlers/sam_audio_cleanup.py`, change the `handle` signature:

```python
def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
    work_dir: Optional[Path] = None,
) -> dict:
```

Replace the `work_dir = Path(tempfile.mkdtemp(...))` line (currently :364) with:

```python
    if work_dir is not None:
        work_dir = Path(work_dir) / f"sam_handler_{job.get('id', 'x')}"
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Legacy path: caller must remove output_file's parent tree after use.
        work_dir = Path(tempfile.mkdtemp(prefix=f"sam_audio_job_{job.get('id', 'x')}_"))
```

Replace the stale trailing comment block in `handle`'s `finally` (the `archive_path` lines ending with `# temp directory cleanup is handled by worker.py after upload`) with:

```python
        # The result ZIP stays in work_dir until the caller consumes it.
        # Callers that passed work_dir own its cleanup; legacy callers must
        # remove output_file's parent tree themselves.
```

In `worker/worker.py`, after `handler_params = set(...)` (:237-239), add:

```python
        if "work_dir" in handler_params:
            handler_kwargs["work_dir"] = work_dir
```

In `streamlit_app.py` `worker_run`, add `work_dir=work_dir` to the `handle(...)` call (the `with tempfile.TemporaryDirectory` block already defines `work_dir = Path(td)`).

- [x] **Step 4: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: new test PASSES; all 42 pre-existing tests still pass.

- [x] **Step 5: Extend the worker test to cover work_dir passing**

In `tests/test_worker_core.py`, add (mirroring `test_process_job_success`'s stub style):

```python
def test_process_job_passes_work_dir(tmp_path):
    received = {}

    def fake_handler(input_path, input_data, job, progress_cb=None, work_dir=None):
        received["work_dir"] = work_dir
        return {"output_data": {"ok": True}, "output_file": None}

    cfg = make_cfg(tmp_path)  # reuse the module's existing cfg builder/fixture
    client = DummyClient()
    job = {"id": 7, "type": "sam_audio_cleanup", "input_data": "{}", "input_filename": ""}
    with patch.object(w, "get_handler", return_value=fake_handler):
        w.process_job(client, cfg, job)
    assert received["work_dir"] is not None
    assert str(received["work_dir"]).startswith(str(cfg.temp_dir))
```

(Adapt `make_cfg`/`DummyClient` names to what `test_worker_core.py` actually defines — read the file first; it stubs `get_handler` via `patch.object(w, ...)` at :76-77.)

- [x] **Step 6: Run full suite again — all green**

Run: `.venv/bin/python -m pytest tests/ -v`

- [x] **Step 7: Commit**

```bash
git add worker/handlers/sam_audio_cleanup.py worker/worker.py streamlit_app.py tests/test_handler_workdir.py tests/test_worker_core.py
git commit -m "fix: stop leaking handler work dirs; caller-owned work_dir param"
```

### Task 2: Streamlit rerun compat shim

sam venv pins Streamlit 1.19.0 where `st.experimental_rerun()` works but is deprecated (removed in ≥1.37, and cortex runs 1.36). Shim it so a future venv upgrade can't silently break the harness.

**Files:**
- Modify: `streamlit_app.py:336-339`

- [x] **Step 1: Add the shim and use it**

Above `main()` in `streamlit_app.py`:

```python
def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
```

Replace `st.experimental_rerun()` (:339) with `_rerun()`.

- [x] **Step 2: Smoke test**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -c "import ast; ast.parse(open('streamlit_app.py').read())" && .venv/bin/python -m pytest tests/ -q`
Expected: parses, suite green. (Full UI smoke happens in Task 9.)

- [x] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "fix: rerun compat shim for streamlit >=1.27 removal of experimental_rerun"
```

### Task 3: Correctness review scan (subagent) + fix confirmed findings

**Files:**
- Review scope: `run_sam_interactive.py`, `worker/worker.py`, `sam_audio_local/loader.py`, `worker/handlers/sam_audio_cleanup.py`
- Fixes + tests: as findings dictate (each its own commit)

- [x] **Step 1: Dispatch review subagent (model: sonnet)**

Prompt packet: repo `/home/longboardfella/sam-audio`; objective "correctness-only review of the four files above: error paths, cancellation propagation, resource cleanup, fp16/fp32 dtype boundaries, chunk merge/crossfade math, OOM-retry ladder consistency"; out of scope: style, refactors, dependency upgrades; evidence format: file:line, one-sentence defect, concrete failure scenario; stop if a file doesn't match this description.

- [x] **Step 2: Fable triages findings**

Reject speculative/no-failure-scenario findings. For each CONFIRMED finding, verify by reopening the cited lines.

- [x] **Step 3: For each confirmed finding: failing test (where testable) → fix → suite green → commit**

One commit per finding: `fix: <finding summary>`. If zero findings survive triage, record "Task 3: no confirmed findings" in the capacity-results doc (Task 5) and move on.

### Task 4: Dependency currency review (record, don't chase)

**Files:**
- Create: `docs/2026-08-01-review-notes.md`

- [x] **Step 1: Collect versions and known issues**

Run: `.venv/bin/pip list --outdated --format=columns | head -40`
Record in `docs/2026-08-01-review-notes.md`: current pins (torch 2.6.0+cu124, streamlit 1.19.0, soundfile 0.13.1, numpy 1.26.4, requests 2.32.5), what's outdated, and the decision: **no torch/streamlit upgrades** (working fp16 build; Turing support in newer torch wheels must be re-verified before any future bump). Note any security-relevant advisories seen for requests/streamlit versions.

- [x] **Step 2: Commit**

```bash
git add docs/2026-08-01-review-notes.md
git commit -m "docs: dependency currency review notes (hold torch/streamlit pins)"
```

---

## Phase 1 — 45-minute capacity gate

### Task 5: Capacity soak on a real 45-min voice file

**Files:**
- Create: `docs/2026-08-01-capacity-soak-results.md`
- Scratch: `$SCRATCH/soak_45min.wav`, `$SCRATCH/soak_monitor.csv`, `$SCRATCH/soak_work/` (all reboot-disposable)

- [x] **Step 1: Build the 45-min test file** (Apollo13.wav is 57.6s of noisy radio voice — ideal content)

```bash
SCRATCH=/tmp/claude-1000/-home-longboardfella-sam-audio/456cf002-c7df-49ad-9c48-620a8286549b/scratchpad
mkdir -p "$SCRATCH"
ffmpeg -y -stream_loop 46 -i /home/longboardfella/sam-audio/Apollo13.wav -c copy "$SCRATCH/soak_45min.wav"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$SCRATCH/soak_45min.wav"
```
Expected duration: ~2708s (~45.1 min).

- [x] **Step 2: Start the resource monitor**

```bash
( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits),$(free -m | awk '/Mem:/{print $3}')" >> "$SCRATCH/soak_monitor.csv"; sleep 10; done ) &
echo $! > "$SCRATCH/soak_mon.pid"
```

- [x] **Step 3: Run the soak** (direct `handle()` call; CLI doesn't exist yet)

```bash
cd /home/longboardfella/sam-audio && .venv/bin/python - <<'EOF'
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/longboardfella/sam-audio")
from worker.handlers.sam_audio_cleanup import handle

SCRATCH = Path("/tmp/claude-1000/-home-longboardfella-sam-audio/456cf002-c7df-49ad-9c48-620a8286549b/scratchpad")
work = SCRATCH / "soak_work"; work.mkdir(parents=True, exist_ok=True)
t0 = time.time()
res = handle(
    input_path=SCRATCH / "soak_45min.wav",
    input_data={"description": "a man speaking over a radio", "convert_to_mono": True,
                "chunk_duration": 60, "overlap": 2.0},
    job={"id": 0, "input_filename": "soak_45min.wav"},
    progress_cb=lambda p, m, s=None: print(f"{time.strftime('%H:%M:%S')} {p:5.1f}% [{s}] {m}", flush=True),
    work_dir=work,
)
print("ELAPSED_SECONDS", round(time.time() - t0, 1))
print(json.dumps(res["output_data"], indent=2))
print("zip:", res["output_file"])
EOF
```

Run this in the background (`run_in_background`) — it will take tens of minutes. Poll progress from its output.

- [x] **Step 4: Stop monitor, evaluate against gate criteria**

```bash
kill "$(cat "$SCRATCH/soak_mon.pid")"
sort -t, -k2 -n "$SCRATCH/soak_monitor.csv" | tail -1   # peak VRAM row
sort -t, -k3 -n "$SCRATCH/soak_monitor.csv" | tail -1   # peak RAM row
```

**Gate (all must hold):** job completes; `options_applied.auto_profile == "requested"` (OOM ladder never engaged); peak VRAM < 44000 MiB total (leaving headroom vs 46080); peak used RAM < 20000 MiB; memory flat across chunks (no monotonic climb in the CSV middle section).

- [x] **Step 5: Record results and commit**

Write `docs/2026-08-01-capacity-soak-results.md`: runtime, realtime factor, peak VRAM/RAM, auto_profile, chunk count, and Task 3 outcome note. Commit:

```bash
git add docs/2026-08-01-capacity-soak-results.md
git commit -m "docs: 45-min capacity soak results (RTX 8000, gate passed)"
```

- [x] **Step 6: USER CHECK (Paul):** listen to `soak_work/.../target.wav` start/middle/end — voice clean, background in residual. Copy the target somewhere persistent first if a reboot is imminent: `cp <zip> ~/sam-audio/audio_output/soak_result.zip` (audio_output/ is gitignored).

---

## Phase 2 — `clean_cli.py`

### Task 6: CLI core — args, status.json, progress protocol, handler invocation

**Files:**
- Create: `clean_cli.py` (repo root)
- Test: `tests/test_clean_cli.py`

**Interfaces:**
- Consumes: `handle(..., work_dir=...)` from Task 1.
- Produces (relied on by Tasks 10, 11, 13):
  - Invocation: `.venv/bin/python clean_cli.py --input <file> --job-json <json> --out-dir <dir> [--opus] [--notify-discord] [--gpu-lock-timeout N]`
  - stdout: one JSON object per line: `{"type":"progress","pct":<int>,"stage":<str>,"message":<str>,"ts":"HH:MM:SS"}`
  - `<out-dir>/status.json`: `{"state":"running","pid":N,"started_at":iso}` → `{"state":"done","metadata":{...},"finished_at":iso}` | `{"state":"error","error":str,"finished_at":iso}`
  - `<out-dir>/result.zip` (target.wav, residual.wav, metadata.json), `<out-dir>/target.ogg` when `--opus`
  - Exit code 0 success / 1 failure. Webhook URL only via env `DISCORD_WEBHOOK_URL`.

- [x] **Step 1: Write failing tests**

Create `tests/test_clean_cli.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import clean_cli


class CleanCliCoreTests(unittest.TestCase):
    def _run(self, td, extra_args=None, fake_handle=None):
        td = Path(td)
        inp = td / "in.wav"; inp.write_bytes(b"RIFFfake")
        jj = td / "job.json"; jj.write_text(json.dumps({"description": "speech"}))
        out = td / "out"
        argv = ["--input", str(inp), "--job-json", str(jj), "--out-dir", str(out)]
        argv += (extra_args or [])

        def default_fake(input_path, input_data, job, progress_cb=None,
                         is_cancelled_cb=None, work_dir=None):
            progress_cb(50, "half way", "separate")
            zip_path = Path(work_dir) / "result_inner.zip"
            import zipfile
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("target.wav", b"x"); zf.writestr("residual.wav", b"y")
            return {"output_data": {"description": "speech", "duration_seconds": 1.0},
                    "output_file": zip_path}

        with patch.object(clean_cli, "handle", side_effect=fake_handle or default_fake):
            rc = clean_cli.main(argv)
        return rc, out

    def test_success_writes_done_status_and_result_zip(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td)
            self.assertEqual(rc, 0)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "done")
            self.assertTrue((out / "result.zip").exists())

    def test_handler_error_writes_error_status_and_rc1(self):
        def boom(**kw):
            raise RuntimeError("separation exploded")
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, fake_handle=boom)
            self.assertEqual(rc, 1)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "error")
            self.assertIn("separation exploded", status["error"])


class ProgressProtocolTests(unittest.TestCase):
    def test_emit_writes_json_line(self):
        import io
        buf = io.StringIO()
        clean_cli.emit(42, "separate", "chunk 3/45", stream=buf)
        line = json.loads(buf.getvalue().strip())
        self.assertEqual(line["type"], "progress")
        self.assertEqual(line["pct"], 42)
        self.assertEqual(line["stage"], "separate")


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run tests — verify import failure**

Run: `.venv/bin/python -m pytest tests/test_clean_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: clean_cli`

- [x] **Step 3: Implement `clean_cli.py`**

```python
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
```

- [x] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_clean_cli.py -v`
Expected: PASS (note: importing `clean_cli` imports the handler module → torch; first run takes a few seconds).

- [x] **Step 5: Commit**

```bash
git add clean_cli.py tests/test_clean_cli.py
git commit -m "feat: clean_cli.py — shared CLI over sam_audio_cleanup with status/progress protocol"
```

### Task 7: Opus + webhook unit tests

**Files:**
- Test: `tests/test_clean_cli.py` (extend)

- [x] **Step 1: Add tests**

```python
import io
import numpy as np
import shutil as _shutil
import soundfile as sf
from unittest.mock import MagicMock


@unittest.skipUnless(_shutil.which("ffmpeg"), "ffmpeg not on PATH")
class OpusEncodeTests(unittest.TestCase):
    def test_encodes_small_wav_to_ogg(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            wav = td / "t.wav"
            sf.write(wav, np.random.uniform(-0.3, 0.3, 16000).astype(np.float32), 16000)
            ogg = clean_cli.encode_opus(wav, td / "t.ogg")
            self.assertTrue(ogg.exists())
            self.assertGreater(ogg.stat().st_size, 0)


class WebhookTests(unittest.TestCase):
    def test_post_success(self):
        fake = MagicMock(status_code=204)
        with patch("requests.post", return_value=fake) as p:
            ok = clean_cli.post_discord_webhook("https://discord/hook", "hello")
        self.assertTrue(ok)
        self.assertEqual(p.call_args.kwargs["json"]["content"], "hello")

    def test_post_network_error_returns_false(self):
        with patch("requests.post", side_effect=OSError("net down")):
            ok = clean_cli.post_discord_webhook("https://discord/hook", "hello")
        self.assertFalse(ok)
```

- [x] **Step 2: Run** `.venv/bin/python -m pytest tests/test_clean_cli.py -v` — all PASS.

- [x] **Step 3: Commit**

```bash
git add tests/test_clean_cli.py
git commit -m "test: opus encode + discord webhook units for clean_cli"
```

### Task 8: GPU lock test

**Files:**
- Test: `tests/test_clean_cli.py` (extend)

- [x] **Step 1: Add test** (flock is per open-file-description, so a second `open()` in the same process contends):

```python
import fcntl


class GpuLockTests(unittest.TestCase):
    def test_lock_blocks_second_acquirer(self):
        with clean_cli.gpu_lock(timeout_s=5):
            fh2 = open(clean_cli.GPU_LOCK_PATH, "w")
            with self.assertRaises(BlockingIOError):
                fcntl.flock(fh2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh2.close()
        # released now: acquire must succeed
        fh3 = open(clean_cli.GPU_LOCK_PATH, "w")
        fcntl.flock(fh3, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh3, fcntl.LOCK_UN)
        fh3.close()
```

- [x] **Step 2: Run** the file's tests — PASS. Run full suite: `.venv/bin/python -m pytest tests/ -q` — green.

- [x] **Step 3: Commit**

```bash
git add tests/test_clean_cli.py
git commit -m "test: gpu flock serialization for clean_cli"
```

### Task 9: CLI real-pipeline smoke (GPU)

- [x] **Step 1: Run a real 60s job through the CLI**

```bash
SCRATCH=/tmp/claude-1000/-home-longboardfella-sam-audio/456cf002-c7df-49ad-9c48-620a8286549b/scratchpad
mkdir -p "$SCRATCH/cli_smoke"
printf '{"description": "a man speaking over a radio", "convert_to_mono": true, "chunk_duration": 60, "overlap": 2.0}' > "$SCRATCH/cli_smoke/job.json"
cd /home/longboardfella/sam-audio
.venv/bin/python clean_cli.py --input Apollo13.wav --job-json "$SCRATCH/cli_smoke/job.json" --out-dir "$SCRATCH/cli_smoke/out" --opus
echo "rc=$?"
```

Expected: progress JSON lines; rc=0; `out/status.json` state=done; `out/result.zip` and `out/target.ogg` exist; `ffprobe out/target.ogg` shows opus mono.

- [x] **Step 2: Streamlit harness smoke** (verifies Task 1+2 changes in the UI path)

Run `streamlit run streamlit_app.py`, process Apollo13.wav with trial_seconds=30, confirm completion and that no new `sam_audio_job_*` dirs appear in `/tmp` afterwards: `ls -d /tmp/sam_audio_job_* 2>/dev/null` → empty.

- [x] **Step 3: Update README and commit**

Add a "clean_cli.py" section to `README.md` (invocation, JSON-line protocol, status.json states, --opus, --notify-discord + `DISCORD_WEBHOOK_URL`).

```bash
git add README.md
git commit -m "docs: clean_cli usage"
```

---

## Phase 3 — Cortex Suite page

### Task 10: `pages/21_Audio_Cleanup.py` in cortex_suite

**Files:**
- Create: `/home/longboardfella/cortex_suite/pages/21_Audio_Cleanup.py`
- Modify: `cortex_engine/version_config.py` (minor feature bump) + version sync workflow

**Interfaces:**
- Consumes: `clean_cli.py` invocation + JSON-line protocol + `status.json`/`result.zip` from Task 6. `SAM_AUDIO_ROOT` env override, default `/home/longboardfella/sam-audio`.

- [x] **Step 1: Write the page**

```python
"""Audio Cleanup — SAM-Audio voice separation via the sam-audio project.

Spawns sam-audio's clean_cli.py in its OWN venv (no torch/SAM deps in the
cortex venv). Progress arrives as JSON lines on the subprocess stdout.
"""
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import streamlit as st

from cortex_engine.utils import get_logger
from cortex_engine.version_config import VERSION_STRING

PAGE_VERSION = VERSION_STRING
SAM_ROOT = Path(os.environ.get("SAM_AUDIO_ROOT", "/home/longboardfella/sam-audio"))
SAM_PYTHON = SAM_ROOT / ".venv" / "bin" / "python"
CLEAN_CLI = SAM_ROOT / "clean_cli.py"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

st.set_page_config(page_title="Audio Cleanup", layout="wide", page_icon="🎙️")
logger = get_logger(__name__)


def init_state():
    ss = st.session_state
    ss.setdefault("ac_proc", None)
    ss.setdefault("ac_events", queue.Queue())
    ss.setdefault("ac_lines", [])
    ss.setdefault("ac_pct", 0)
    ss.setdefault("ac_status", "Idle")
    ss.setdefault("ac_out_dir", None)
    ss.setdefault("ac_error", None)
    ss.setdefault("ac_running", False)


def reader_thread(proc, events):
    try:
        for raw in iter(proc.stdout.readline, ""):
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.put(json.loads(raw))
            except json.JSONDecodeError:
                events.put({"type": "log", "message": raw})
        proc.wait()
    finally:
        events.put({"type": "exit", "rc": proc.returncode})


def start_job(upload, description, opts):
    work = Path(tempfile.mkdtemp(prefix="cortex_audio_"))
    input_path = work / upload.name
    input_path.write_bytes(upload.getvalue())
    payload = {
        "description": description or "speech",
        "convert_to_mono": True,
        "chunk_duration": int(opts["chunk_duration"]),
        "overlap": float(opts["overlap"]),
        "loudness_normalize": bool(opts["loudness"]),
        "trial_seconds": int(opts["trial_seconds"]),
        "rerank": 1,
        "predict_spans": False,
        "device": "auto",
        "memory_fraction": 0.85,
        "allow_cpu_fallback": True,
    }
    job_json = work / "job.json"
    job_json.write_text(json.dumps(payload))
    out_dir = work / "out"
    proc = subprocess.Popen(
        [str(SAM_PYTHON), str(CLEAN_CLI), "--input", str(input_path),
         "--job-json", str(job_json), "--out-dir", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(SAM_ROOT),
    )
    st.session_state.ac_proc = proc
    st.session_state.ac_out_dir = out_dir
    st.session_state.ac_running = True
    st.session_state.ac_error = None
    st.session_state.ac_pct = 0
    st.session_state.ac_lines = []
    st.session_state.ac_status = "Starting SAM-Audio job"
    threading.Thread(target=reader_thread,
                     args=(proc, st.session_state.ac_events), daemon=True).start()


def drain_events():
    q = st.session_state.ac_events
    while True:
        try:
            evt = q.get_nowait()
        except queue.Empty:
            return
        if evt.get("type") == "progress":
            st.session_state.ac_pct = max(0, min(100, int(evt.get("pct", 0))))
            msg = f"[{evt.get('stage')}] {evt.get('message')}"
            st.session_state.ac_status = msg
            st.session_state.ac_lines.append(f"{evt.get('ts')} | {evt.get('pct'):>3}% | {msg}")
            st.session_state.ac_lines = st.session_state.ac_lines[-200:]
        elif evt.get("type") == "log":
            st.session_state.ac_lines.append(str(evt.get("message")))
        elif evt.get("type") == "exit":
            st.session_state.ac_running = False
            if evt.get("rc") != 0:
                out_dir = st.session_state.ac_out_dir
                err = "clean_cli exited with an error"
                try:
                    status = json.loads((Path(out_dir) / "status.json").read_text())
                    err = status.get("error", err)
                except Exception:
                    pass
                st.session_state.ac_error = err


def main():
    init_state()
    drain_events()

    st.title("🎙️ Audio Cleanup")
    st.caption(f"SAM-Audio voice separation — v{PAGE_VERSION}. "
               "Describe the sound to EXTRACT (e.g. 'a man speaking over a radio'); "
               "everything else lands in residual.wav.")

    if not SAM_PYTHON.exists():
        st.error(f"sam-audio venv not found at {SAM_PYTHON}. Set SAM_AUDIO_ROOT.")
        return

    upload = st.file_uploader(
        "Audio or video file",
        type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "mp4", "mkv", "mov"])
    description = st.text_input("What to extract", value="speech")

    with st.expander("Advanced options"):
        chunk_duration = st.number_input("Chunk duration (s)", 5, 600, 60, 5)
        overlap = st.number_input("Chunk overlap (s)", 0.0, 30.0, 2.0, 0.5)
        loudness = st.checkbox("Loudness-normalize target (-16 LUFS)", value=True)
        trial_seconds = st.number_input("Trial only first N seconds (0 = full)", 0, 86400, 0, 5)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Clean Audio", type="primary",
                     disabled=st.session_state.ac_running or upload is None):
            if upload.size > MAX_UPLOAD_BYTES:
                st.error("File exceeds 2GB limit")
            else:
                start_job(upload, description,
                          {"chunk_duration": chunk_duration, "overlap": overlap,
                           "loudness": loudness, "trial_seconds": trial_seconds})
                st.rerun()
    with col2:
        if st.session_state.ac_running and st.button("Stop"):
            proc = st.session_state.ac_proc
            if proc and proc.poll() is None:
                proc.terminate()

    st.progress(st.session_state.ac_pct)
    st.caption(st.session_state.ac_status)
    if st.session_state.ac_lines:
        st.code("\n".join(st.session_state.ac_lines[-30:]), language="text")

    if st.session_state.ac_error:
        st.error(st.session_state.ac_error)

    out_dir = st.session_state.ac_out_dir
    if (not st.session_state.ac_running) and out_dir:
        zip_path = Path(out_dir) / "result.zip"
        if zip_path.exists():
            zip_bytes = zip_path.read_bytes()
            st.download_button("Download ZIP (target + residual + metadata)",
                               data=zip_bytes, file_name="audio_cleanup_result.zip",
                               mime="application/zip")
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Cleaned voice (target)**")
                    if "target.wav" in names:
                        st.audio(zf.read("target.wav"), format="audio/wav")
                with c2:
                    st.markdown("**Removed background (residual)**")
                    if "residual.wav" in names:
                        st.audio(zf.read("residual.wav"), format="audio/wav")

    if st.session_state.ac_running:
        time.sleep(1)
        st.rerun()


main()
```

- [x] **Step 2: Syntax check + manual smoke**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -c "import ast; ast.parse(open('pages/21_Audio_Cleanup.py').read())"`
Then `venv/bin/streamlit run Cortex_Suite.py`, open the Audio Cleanup page, run Apollo13.wav with trial_seconds=30. Expected: progress bar advances, players render, ZIP downloads.

- [x] **Step 3: Cortex version workflow** (per cortex CLAUDE.md — read `cortex_engine/version_config.py` first, apply a minor feature bump):

```bash
cd /home/longboardfella/cortex_suite
# edit cortex_engine/version_config.py: bump CORTEX_VERSION minor + VERSION_METADATA entry
venv/bin/python scripts/version_manager.py --sync-all
venv/bin/python scripts/version_manager.py --update-changelog
venv/bin/python scripts/version_manager.py --check
```

- [x] **Step 4: Commit (cortex repo)**

Review `git status` / `git diff --stat` first — the cortex working tree carries unrelated dirt; stage ONLY the page, version_config.py, CHANGELOG.md, and files version-sync actually touched for this change.

```bash
git add pages/21_Audio_Cleanup.py cortex_engine/version_config.py CHANGELOG.md   # + sync-touched files after review
git commit -m "feat: Audio Cleanup page — SAM-Audio voice separation via sam-audio clean_cli"
```

---

## Phase 4 — Bridge endpoints + Discord webhook

### Task 11: `/audio-clean` + `/audio-status` in kb-query-server.py

**Files:**
- Backup then modify: `/home/longboardfella/kb-query-server.py` (NOT in git)
- Stub for testing: `$SCRATCH/stub_clean_cli.sh`

**Interfaces:**
- Consumes: CLI invocation contract from Task 6 (spawn via env `KB_AUDIO_CLEAN_CLI`, default `/home/longboardfella/sam-audio/.venv/bin/python /home/longboardfella/sam-audio/clean_cli.py`).
- Produces (relied on by Task 13): `POST /audio-clean?filename=<urlenc>&description=<urlenc>&loudness=1` with raw binary body → `{"ok":true,"job_id":"<id>"}`; `GET /audio-status?id=<job_id>` → status.json content (+ `"state":"stale"` if pid dead, `"state":"submitted"` if no status yet, `"ok":false,"error":"unknown job"` otherwise). Spool: `~/sam-audio/spool/<job_id>/`.

- [x] **Step 1: Backup the live script**

```bash
cp /home/longboardfella/kb-query-server.py /home/longboardfella/kb-query-server.py.bak-20260801
```

- [x] **Step 2: Add constants + helpers** (near the existing `TOKEN`/`BIND`/`PORT` block, kb-query-server.py:39-43; `import secrets, shutil, subprocess` and `from pathlib import Path` at the top if absent):

```python
SPOOL_DIR = Path(os.path.expanduser("~/sam-audio/spool"))
AUDIO_CLEAN_CLI = os.environ.get(
    "KB_AUDIO_CLEAN_CLI",
    "/home/longboardfella/sam-audio/.venv/bin/python /home/longboardfella/sam-audio/clean_cli.py",
)
AUDIO_WEBHOOK_FILE = os.path.expanduser("~/.discord-audio-webhook")
MAX_AUDIO_UPLOAD = 2 * 1024 ** 3
AUDIO_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mkv", ".mov"}
AUDIO_SPOOL_KEEP_DAYS = 7


def _purge_spool(days: int = AUDIO_SPOOL_KEEP_DAYS) -> None:
    if not SPOOL_DIR.exists():
        return
    cutoff = time.time() - days * 86400
    for d in SPOOL_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def audio_clean_spawn(job_dir: Path, input_path: Path, description: str, loudness: bool) -> None:
    payload = {
        "description": description or "speech", "convert_to_mono": True,
        "chunk_duration": 60, "overlap": 2.0,
        "loudness_normalize": bool(loudness), "rerank": 1, "predict_spans": False,
        "device": "auto", "memory_fraction": 0.85, "allow_cpu_fallback": True,
    }
    (job_dir / "job.json").write_text(json.dumps(payload))
    env = dict(os.environ)
    try:
        env["DISCORD_WEBHOOK_URL"] = open(AUDIO_WEBHOOK_FILE).read().strip()
    except OSError:
        pass
    cmd = AUDIO_CLEAN_CLI.split() + [
        "--input", str(input_path), "--job-json", str(job_dir / "job.json"),
        "--out-dir", str(job_dir), "--opus", "--notify-discord",
    ]
    log = open(job_dir / "job.log", "ab")
    subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                     start_new_session=True, env=env,
                     cwd="/home/longboardfella/sam-audio")


def audio_clean_status(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f\-]{8,64}", job_id or ""):
        return {"ok": False, "error": "bad job id"}
    job_dir = SPOOL_DIR / job_id
    if not job_dir.is_dir():
        return {"ok": False, "error": "unknown job"}
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return {"ok": True, "job_id": job_id, "state": "submitted"}
    status = json.loads(status_path.read_text())
    if status.get("state") == "running" and not _pid_alive(status.get("pid", -1)):
        status["state"] = "stale"
    return {"ok": True, "job_id": job_id, **status}
```

- [x] **Step 3: Add a streamed raw-body reader to `Handler`** (next to `_read_json`, :2001):

```python
    def _read_body_to_file(self, dest, max_bytes: int) -> int:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise ValueError("empty body")
        if n > max_bytes:
            raise ValueError(f"body too large: {n} bytes (cap {max_bytes})")
        remaining = n
        with open(dest, "wb") as f:
            while remaining:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError(f"truncated body: got {n - remaining} of {n} bytes")
                f.write(chunk)
                remaining -= len(chunk)
        return n
```

- [x] **Step 4: Add dispatch branches in `_handle()`** (alongside the existing `/add` branch, following its style):

```python
        if parsed.path.rstrip("/") == "/audio-clean" and self.command == "POST":
            qs = parse_qs(parsed.query)
            try:
                filename = (qs.get("filename") or ["input.wav"])[0]
                ext = os.path.splitext(filename)[1].lower()
                if ext not in AUDIO_ALLOWED_EXT:
                    return self._reply(400, {"ok": False, "error": f"unsupported extension: {ext}"})
                job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4)
                job_dir = SPOOL_DIR / job_id
                job_dir.mkdir(parents=True, exist_ok=False)
                input_path = job_dir / f"input{ext}"
                self._read_body_to_file(input_path, MAX_AUDIO_UPLOAD)
                audio_clean_spawn(
                    job_dir, input_path,
                    description=(qs.get("description") or ["speech"])[0],
                    loudness=(qs.get("loudness") or ["1"])[0] not in ("0", "false", ""),
                )
                return self._reply(200, {"ok": True, "job_id": job_id})
            except ValueError as e:
                return self._reply(400, {"ok": False, "error": str(e)})
            except Exception as e:  # noqa: BLE001
                return self._reply(500, {"ok": False, "error": str(e)})

        if parsed.path.rstrip("/") == "/audio-status":
            qs = parse_qs(parsed.query)
            try:
                return self._reply(200, audio_clean_status((qs.get("id") or [""])[0]))
            except Exception as e:  # noqa: BLE001
                return self._reply(500, {"ok": False, "error": str(e)})
```

Note: job_id regex must accept the `YYYYMMDD-HHMMSS-hex` form (digits included) — `[0-9a-f\-]` does. Also add `_purge_spool()` in `__main__` before `serve_forever()`.

- [x] **Step 5: Integration test on a private port with a stub CLI**

Create `$SCRATCH/stub_clean_cli.sh`:

```bash
#!/usr/bin/env bash
# Stub for clean_cli: consumes the same args, writes a done status.
OUT=""
while [ $# -gt 0 ]; do case "$1" in --out-dir) OUT="$2"; shift 2;; *) shift;; esac; done
mkdir -p "$OUT"
echo '{"state":"done","metadata":{"stub":true}}' > "$OUT/status.json"
echo stub-zip > "$OUT/result.zip"
```

```bash
chmod +x "$SCRATCH/stub_clean_cli.sh"
cd /home/longboardfella
KB_QUERY_TOKEN=testtoken KB_QUERY_BIND=127.0.0.1 KB_QUERY_PORT=7999 \
  KB_AUDIO_CLEAN_CLI="$SCRATCH/stub_clean_cli.sh" \
  setsid nohup ./kb-query-server.py > "$SCRATCH/test-bridge.log" 2>&1 &
sleep 8   # warm-up
# submit
curl -sS -X POST -H "Authorization: Bearer testtoken" \
  --data-binary @/home/longboardfella/sam-audio/Apollo13.wav \
  "http://127.0.0.1:7999/audio-clean?filename=Apollo13.wav&description=radio%20voice"
# → {"ok": true, "job_id": "..."} ; then with that id:
curl -sS -H "Authorization: Bearer testtoken" "http://127.0.0.1:7999/audio-status?id=<JOB_ID>"
# → {"ok": true, ..., "state": "done", "metadata": {"stub": true}}
# negative: bad token → 401; bad ext → 400
curl -sS -o /dev/null -w "%{http_code}\n" -X POST -H "Authorization: Bearer wrong" \
  --data-binary "x" "http://127.0.0.1:7999/audio-clean?filename=a.wav"     # 401
curl -sS -X POST -H "Authorization: Bearer testtoken" \
  --data-binary "x" "http://127.0.0.1:7999/audio-clean?filename=a.exe"     # unsupported extension
# check spool layout
ls ~/sam-audio/spool/<JOB_ID>/   # input.wav job.json job.log result.zip status.json
# kill the test instance
pkill -f "KB_QUERY_PORT=7999" || kill %1
```

(Adjust the final kill to target the test server's pid — capture `$!` at launch.)

- [x] **Step 6: Restart the LIVE bridge and verify it still serves**

```bash
pkill -f kb-query-server.py; sleep 2
bash /home/longboardfella/kb-query-server-ensure.sh
sleep 10
TOKEN=$(cat ~/.kb-query-token)
curl -sS -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" \
  "http://100.118.92.17:7333/audio-status?id=nope"   # expect 200 ({"ok":false,...})
tail -5 /tmp/kb-query-server.log
```

Expected: server listening, existing endpoints unaffected (spot-check one: an authed GET that previously worked, e.g. `/graph` per its contract).

- [x] **Step 7: Record the change** — append a dated section to `/home/longboardfella/nemoclaw_ops/docs/` (new file `2026-08-01-kb-bridge-audio-clean-endpoints.md`) describing the two endpoints, spool layout, env knobs (`KB_AUDIO_CLEAN_CLI`, `~/.discord-audio-webhook`), and the `.bak-20260801` backup. Commit in nemoclaw_ops if it's a git repo (check `git -C ~/nemoclaw_ops status`), else leave the file.

### Task 12: Discord webhook config + real short job via bridge

- [x] **Step 1: USER STEP (Paul):** create a webhook in the target Discord channel (Channel settings → Integrations → Webhooks → New Webhook → Copy URL) and provide the URL.

- [x] **Step 2: Store it**

```bash
printf '%s' '<WEBHOOK_URL>' > ~/.discord-audio-webhook
chmod 600 ~/.discord-audio-webhook
```

- [x] **Step 3: Webhook sanity test** (no GPU):

```bash
curl -sS -F 'content=SAM-Audio webhook test' "$(cat ~/.discord-audio-webhook)"
```
Expected: message appears in the channel.

- [x] **Step 4: Real end-to-end short job through the live bridge**

```bash
TOKEN=$(cat ~/.kb-query-token)
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  --data-binary @/home/longboardfella/sam-audio/Apollo13.wav \
  "http://100.118.92.17:7333/audio-clean?filename=Apollo13.wav&description=a%20man%20speaking%20over%20a%20radio"
```
Poll `/audio-status?id=...` until `done`. Expected: `target.ogg` posted to the Discord channel with the ✅ summary within a few minutes.

---

## Phase 5 — sp4 wrapper + Hermes registration

### Task 13: `kb-audio-clean` on sp4

**Files:**
- Create on sp4: `/home/paul/.local/bin/kb-audio-clean`
- Modify on sp4: Hermes tool/prompt registration (location discovered in Step 1)

- [x] **Step 1: Inspect sp4 conventions** (read-only):

```bash
ssh sp4 -o RemoteCommand=none 'ls ~/.local/bin/kb-* && cat ~/.local/bin/kb-ask 2>/dev/null || cat $(ls ~/.local/bin/kb-* | head -1)'
ssh sp4 -o RemoteCommand=none 'grep -rn "kb-" ~/.hermes/config.yaml | head; ls ~/.hermes/'
```
Learn: where the bearer token lives on sp4, how existing wrappers call the bridge, and how tools are exposed to Hermes (config.yaml tools list vs. prompt snippet).

- [x] **Step 2: Install the wrapper** (align the TOKEN line with the discovered convention before installing):

```bash
#!/usr/bin/env bash
# kb-audio-clean <audio-file> [description...]
# Submits audio to fastfella for SAM-Audio cleanup; result is posted back to
# Discord automatically by the processing host when done.
set -euo pipefail
FILE="${1:?usage: kb-audio-clean <audio-file> [description...]}"; shift || true
DESC="${*:-speech}"
TOKEN="$(cat "$HOME/.kb-query-token")"   # ← adjust to sp4's actual token source
BASE="http://100.118.92.17:7333"
[ -f "$FILE" ] || { echo "no such file: $FILE" >&2; exit 1; }
ENC_DESC=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$DESC")
ENC_FN=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$(basename "$FILE")")
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  --data-binary @"$FILE" \
  "$BASE/audio-clean?filename=$ENC_FN&description=$ENC_DESC"
echo
echo "Submitted. Cleaned audio will be posted to this Discord channel when processing finishes."
```

Deploy: write locally, `scp` to `sp4:/home/paul/.local/bin/kb-audio-clean`, `ssh sp4 -o RemoteCommand=none 'chmod +x ~/.local/bin/kb-audio-clean'`.

- [x] **Step 3: Test from sp4 shell**

```bash
ssh sp4 -o RemoteCommand=none '~/.local/bin/kb-audio-clean <some-small-audio-file-on-sp4> "a man speaking"'
```
Expected: `{"ok":true,"job_id":...}` and, minutes later, the cleaned Opus in Discord.

- [x] **Step 4: Register with Hermes** — following whatever pattern Step 1 revealed (tools list in `~/.hermes/config.yaml` and/or the prompt snippet used for kb-ask): add `kb-audio-clean` with usage guidance:

> When the user posts an audio/voice attachment and asks for cleanup/denoising, run `kb-audio-clean <attachment-path> <description of the sound to KEEP, e.g. "a man speaking">` and tell the user the job is submitted and the cleaned audio will be posted to the channel automatically. Do not wait for completion.

Restart/reload Hermes per its convention if needed. Back up any file edited on sp4 first (`cp X X.bak-20260801`).

---

## Phase 6 — End-to-end + wrap-up

### Task 14: Discord round-trip, docs, memory

- [ ] **Step 1: USER STEP (Paul):** post a real voice recording into the Discord channel and ask SlowClaw to clean it. Expected: Hermes acknowledges → cleaned `target.ogg` + ✅ summary appears in-channel.

- [x] **Step 2: Verify job artifacts on fastfella:** `ls ~/sam-audio/spool/` shows the job; `status.json` = done; `job.log` clean.

- [x] **Step 3: Update sam-audio README** — add "Cortex Suite page" and "Discord (Hermes) loop" sections: architecture sketch, spool location, webhook file, bridge endpoints, sp4 wrapper. Commit: `docs: cortex + discord integration`.

- [x] **Step 4: Update memory** — add/update a memory file describing the finished pipeline (bridge endpoints, spool, wrapper name, webhook file location) so future sessions don't re-derive it.

- [x] **Step 5: Mark all checkboxes in this plan, final commit.**

---

## Self-review notes (author)

- Spec coverage: review fixes (T1-4), capacity gate (T5), CLI (T6-9), Cortex page (T10), bridge (T11-12), sp4 (T13), E2E (T14) — all spec sections mapped.
- Deviations from spec, both approved-in-spirit (simpler, same behavior): raw-body upload instead of multipart (bridge has no multipart parser; matches its JSON/raw style), webhook URL via env not argv (avoids /proc exposure).
- Type consistency: `handle(work_dir=)` used identically in T1/T5/T6; CLI contract in T6 matches usage in T10/T11/T13; `status.json` states (`running/done/error/stale/submitted`) consistent across T6/T11.
