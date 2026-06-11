# Loudness Normalization + Video Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an off-by-default "loudness-normalize target to −16 LUFS / −3 dB true peak" option (Streamlit checkbox + worker payload + smoke-test flag) and accept `.mp4`/`.mkv` inputs everywhere via one ffmpeg audio-extraction helper.

**Architecture:** Loudness normalization is two-pass ffmpeg `loudnorm` (measure → apply one linear gain) implemented in `worker/handlers/sam_audio_cleanup.py` as a post-process step on `target.wav` only; JSON parsing is a pure function for fast tests. Video extraction is one helper in `run_sam_interactive.py` (the worker handler already imports from it) called at intake by both the CLI directory loop and the handler.

**Tech Stack:** Python 3.11 (`.venv/bin/python` ALWAYS), ffmpeg 6.1.1 (system, has `loudnorm`/libx264), pytest 9, soundfile/numpy (already installed).

**Spec:** `docs/superpowers/specs/2026-06-11-loudnorm-video-input-design.md`

**Branch:** create `feature/loudnorm-video-input` from `main` before Task 1.

---

## Background facts (verified — do not re-derive)

- ffmpeg `loudnorm` prints its measurement JSON to **stderr**, after a `[Parsed_loudnorm_0 @ 0x…]` banner line. Keys are strings: `"input_i"`, `"input_tp"`, `"input_lra"`, `"input_thresh"`, `"target_offset"`.
- Handler post-process block lives at `worker/handlers/sam_audio_cleanup.py:446-455` (finds `target_path`/`residual_path`, then peak-normalize, then transcode). Payload parsing is at lines ~243-258 (`_as_bool`, `_clamp_float` helpers exist). `ffmpeg_bin` is already resolved at line ~258. Existing ffmpeg subprocess style: `subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)` + `RuntimeError(f"...: {proc.stderr[-500:]}")` (see `_apply_trial_cut`, line 128).
- Handler intake: `process_input = input_path` then optional trial cut, inside `try:` with `work_dir = Path(tempfile.mkdtemp(...))` (lines ~269-278). Output naming derives from the processed file's stem, so the extracted wav must be named `<original_stem>.wav`.
- `run_sam_interactive.py`: `AUDIO_EXTENSIONS` at line 48; `find_audio_files()` at line 532 globs per-extension; the batch loop calls `process_audio_file(audio_file, ...)` at lines ~1203-1216. The file already imports `tempfile`; CHECK whether it imports `subprocess`, `os`, `shutil`, `Optional` and add any missing ones.
- `streamlit_app.py`: uploader `type=[...]` at line 203; normalize control at ~line 184; payload dict at lines 215-228.
- `colab_smoke_test.py`: argparse flags at lines 30-42 (`--predict-spans` shows the `action="store_true"` pattern), payload dict at lines 56-67.
- Tests are unittest-style, run via `.venv/bin/python -m pytest tests/ -v`. The GPU integration tests are excluded with `-k "not Integration"`. Importing the worker handler imports torch (slow first time) — that's normal.
- Streamlit's `st.file_uploader` `type` list uses bare extensions without dots.

## File Structure

- Modify: `worker/handlers/sam_audio_cleanup.py` — `_parse_loudnorm_json`, `_measure_loudness`, `_apply_loudness_normalize`, payload key, post-process step, video intake.
- Modify: `run_sam_interactive.py` — `VIDEO_EXTENSIONS`, `extract_audio_to_wav`, dir-scan + CLI loop wiring.
- Modify: `streamlit_app.py` — uploader types, loudness checkbox, payload key.
- Modify: `colab_smoke_test.py` — `--loudness-normalize` flag.
- Create: `tests/test_postprocess.py` — loudnorm JSON parser, extension routing, ffmpeg round-trips.
- Modify: `README.md` — short feature notes.

---

### Task 1: `_parse_loudnorm_json` (TDD, fast)

**Files:**
- Modify: `worker/handlers/sam_audio_cleanup.py` (add function near `_apply_peak_normalize`, ~line 156)
- Create: `tests/test_postprocess.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_postprocess.py`:

```python
from __future__ import annotations

import unittest

from worker.handlers.sam_audio_cleanup import _parse_loudnorm_json

SAMPLE_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023
  Stream #0:0: Audio: pcm_s16le, 48000 Hz, mono, s16
[Parsed_loudnorm_0 @ 0x5b7e8c] 
{
    "input_i" : "-27.61",
    "input_tp" : "-4.47",
    "input_lra" : "6.30",
    "input_thresh" : "-38.21",
    "output_i" : "-16.58",
    "output_tp" : "-3.00",
    "output_lra" : "5.90",
    "output_thresh" : "-27.07",
    "normalization_type" : "dynamic",
    "target_offset" : "0.58"
}
"""


class ParseLoudnormJsonTests(unittest.TestCase):
    def test_parses_measured_values(self):
        m = _parse_loudnorm_json(SAMPLE_STDERR)
        self.assertEqual(m["input_i"], "-27.61")
        self.assertEqual(m["input_tp"], "-4.47")
        self.assertEqual(m["input_lra"], "6.30")
        self.assertEqual(m["input_thresh"], "-38.21")
        self.assertEqual(m["target_offset"], "0.58")

    def test_no_json_block_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json("ffmpeg version 6.1.1\nno json here\n")

    def test_malformed_json_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json('prefix\n{\n    "input_i" : "-27.61",\n')

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json('{\n  "input_i" : "-27.61"\n}\n')


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_postprocess.py -v`
Expected: FAIL with `ImportError: cannot import name '_parse_loudnorm_json'`

- [ ] **Step 3: Implement**

In `worker/handlers/sam_audio_cleanup.py`, after `_apply_peak_normalize` (ends line 155), add:

```python
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
```

NOTE: `rfind("{")` would land INSIDE the block if the JSON were followed by more braces — it isn't (the block is the last thing loudnorm prints), but `rfind("{")` with `rfind("}")` handles the malformed-JSON test naturally: `'prefix\n{\n "input_i"...'` has no closing brace → ValueError. `json` is already imported at the top of the handler (verify; add if not).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_postprocess.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/longboardfella/sam-audio
git add worker/handlers/sam_audio_cleanup.py tests/test_postprocess.py
git commit -m "feat: parse ffmpeg loudnorm measurement JSON"
```

---

### Task 2: Two-pass loudnorm + handler wiring + round-trip test

**Files:**
- Modify: `worker/handlers/sam_audio_cleanup.py` (helpers after `_parse_loudnorm_json`; payload parse ~line 246; post-process ~line 449)
- Test: `tests/test_postprocess.py` (append)

- [ ] **Step 1: Append the failing round-trip test**

Append to `tests/test_postprocess.py` (add `import shutil`, `import tempfile`, `from pathlib import Path`, `import numpy as np`, `import soundfile as sf` at the top of the file):

```python
@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class LoudnormRoundTripTests(unittest.TestCase):
    def test_quiet_tone_lands_at_minus_16_lufs(self):
        from worker.handlers.sam_audio_cleanup import (
            _apply_loudness_normalize,
            _measure_loudness,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tone.wav"
            sr = 48000
            t = np.linspace(0, 4.0, 4 * sr, endpoint=False)
            sf.write(path, (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

            _apply_loudness_normalize(path, "ffmpeg")

            measured = _measure_loudness(path, "ffmpeg")
            self.assertAlmostEqual(float(measured["input_i"]), -16.0, delta=1.0)
            self.assertLessEqual(float(measured["input_tp"]), -2.9)
            info = sf.info(str(path))
            self.assertEqual(info.samplerate, sr)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_postprocess.py -v`
Expected: round-trip test FAILS with ImportError (`_apply_loudness_normalize` undefined); the 4 parser tests still pass.

- [ ] **Step 3: Implement the helpers**

In `worker/handlers/sam_audio_cleanup.py`, directly after `_parse_loudnorm_json`, add:

```python
def _measure_loudness(path: Path, ffmpeg_bin: str) -> dict:
    cmd = [
        ffmpeg_bin, "-hide_banner", "-nostats",
        "-i", str(path),
        "-af", f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg loudness measurement failed: {proc.stderr[-500:]}")
    return _parse_loudnorm_json(proc.stderr)


def _apply_loudness_normalize(path: Path, ffmpeg_bin: str) -> None:
    """Two-pass EBU R128 normalization to -16 LUFS / -3 dB true peak.

    linear=true applies a single clean gain (no dynamics processing).
    loudnorm resamples to 192k internally, so pin the original rate back.
    """
    measured = _measure_loudness(path, ffmpeg_bin)
    sample_rate = sf.info(str(path)).samplerate
    out = path.with_name(f"{path.stem}_loudnorm.wav")
    af = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )
    cmd = [ffmpeg_bin, "-y", "-i", str(path), "-af", af, "-ar", str(sample_rate), str(out)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg loudnorm failed: {proc.stderr[-500:]}")
    out.replace(path)
```

- [ ] **Step 4: Wire the payload key and post-process step**

In the payload-parsing section, after `normalize_percent = _clamp_float(payload.get("normalize_percent"), 0.0, 0.0, 100.0)` (line ~246), add:

```python
    loudness_normalize = _as_bool(payload.get("loudness_normalize"), False)
```

In the post-process stage, find (lines ~449-454):

```python
        if normalize_percent > 0:
            _safe_progress(progress_cb, 75, f"Normalizing to {normalize_percent:.1f}% peak", "postprocess")
            _apply_peak_normalize(target_path, normalize_percent)
            _apply_peak_normalize(residual_path, normalize_percent)
            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Cancelled during post-processing")
```

and add immediately after that block:

```python
        if loudness_normalize:
            _safe_progress(progress_cb, 78, "Loudness normalizing target (-16 LUFS, -3 dB TP)", "postprocess")
            _apply_loudness_normalize(target_path, ffmpeg_bin)
            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Cancelled during post-processing")
```

(Target only — residual untouched, per spec. Ordering: loudnorm after peak-normalize so it determines the final target level; the existing transcode step stays after both.)

- [ ] **Step 5: Run tests**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_postprocess.py -v`
Expected: 5 passed.
Then the full fast suite: `.venv/bin/python -m pytest tests/ -v -k "not Integration"`
Expected: 16 passed (11 existing + 5 new), 5 deselected.

- [ ] **Step 6: Commit**

```bash
cd /home/longboardfella/sam-audio
git add worker/handlers/sam_audio_cleanup.py tests/test_postprocess.py
git commit -m "feat: two-pass loudnorm post-process for target.wav behind loudness_normalize payload flag"
```

---

### Task 3: `extract_audio_to_wav` + CLI video support

**Files:**
- Modify: `run_sam_interactive.py` (constants line 48; `find_audio_files` line 532; batch loop ~line 1203)
- Test: `tests/test_postprocess.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_postprocess.py`:

```python
import subprocess


def _make_test_mp4(path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
        "-shortest", "-c:v", "libx264", "-c:a", "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class ExtractAudioTests(unittest.TestCase):
    def test_mp4_extraction(self):
        from run_sam_interactive import extract_audio_to_wav

        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "clip.mp4"
            _make_test_mp4(mp4)
            wav = extract_audio_to_wav(mp4, out_dir=Path(td))
            self.assertEqual(wav.name, "clip.wav")
            info = sf.info(str(wav))
            self.assertEqual(info.samplerate, 48000)
            self.assertEqual(info.channels, 2)
            self.assertAlmostEqual(info.duration, 2.0, delta=0.2)

    def test_mkv_extraction(self):
        from run_sam_interactive import extract_audio_to_wav

        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "clip.mp4"
            _make_test_mp4(mp4)
            mkv = Path(td) / "clip.mkv"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4), "-c", "copy", str(mkv)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            wav = extract_audio_to_wav(mkv, out_dir=Path(td))
            self.assertEqual(wav.name, "clip.wav")
            self.assertAlmostEqual(sf.info(str(wav)).duration, 2.0, delta=0.2)

    def test_no_audio_stream_raises(self):
        from run_sam_interactive import extract_audio_to_wav

        with tempfile.TemporaryDirectory() as td:
            silent = Path(td) / "noaudio.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                 "-c:v", "libx264", str(silent)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            with self.assertRaises(RuntimeError):
                extract_audio_to_wav(silent, out_dir=Path(td))


class VideoExtensionRoutingTests(unittest.TestCase):
    def test_find_audio_files_includes_video(self):
        from run_sam_interactive import find_audio_files

        with tempfile.TemporaryDirectory() as td:
            for name in ("a.wav", "b.mp4", "c.mkv", "d.txt", "e_target.wav"):
                (Path(td) / name).touch()
            found = {p.name for p in find_audio_files(Path(td))}
            self.assertEqual(found, {"a.wav", "b.mp4", "c.mkv"})
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_postprocess.py -v`
Expected: 4 new tests FAIL with ImportError (`extract_audio_to_wav`); previous 5 still pass.

- [ ] **Step 3: Implement in `run_sam_interactive.py`**

At line 48, after `AUDIO_EXTENSIONS = {...}`, add:

```python
VIDEO_EXTENSIONS = {'.mp4', '.mkv'}
```

Check imports at the top of the file: it already imports `tempfile`; ensure `subprocess`, `os`, `shutil`, and `Optional` (from typing) are imported, adding any that are missing.

Near `find_audio_files` (line 532), change the glob loop:

```python
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(input_dir.glob(f'*{ext}'))
```

to:

```python
    for ext in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        audio_files.extend(input_dir.glob(f'*{ext}'))
```

Add the helper (place it right after `find_audio_files`):

```python
def extract_audio_to_wav(path: Path, ffmpeg_bin: str = "ffmpeg", out_dir: Optional[Path] = None) -> Path:
    """Extract the default audio track of a video container to <stem>.wav.

    The wav keeps the source stem so downstream output naming is unchanged.
    When out_dir is None a fresh temp dir is created; the caller owns cleanup
    of the returned file's parent in that case.
    """
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="sam_video_audio_"))
    out = out_dir / f"{path.stem}.wav"
    cmd = [
        ffmpeg_bin, "-y", "-i", str(path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
        str(out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"ffmpeg audio extraction failed for {path.name}: {proc.stderr[-500:]}"
        )
    return out
```

(Note: ffmpeg exits non-zero when there is no audio stream to map with `-vn`, which is what the no-audio test asserts.)

- [ ] **Step 4: Wire the CLI batch loop**

In the batch loop (~line 1203), find:

```python
            processed_files.add(file_id)
            success = process_audio_file(
                audio_file,
```

Replace with:

```python
            processed_files.add(file_id)
            process_path = audio_file
            extracted_dir = None
            if audio_file.suffix.lower() in VIDEO_EXTENSIONS:
                print(f"  Extracting audio track from {audio_file.name} ...")
                process_path = extract_audio_to_wav(audio_file)
                extracted_dir = process_path.parent
            success = process_audio_file(
                process_path,
```

Then find the end of that call plus the success/fail accounting:

```python
            if success:
                success_count += 1
            else:
                fail_count += 1
```

Replace with:

```python
            if extracted_dir is not None:
                shutil.rmtree(extracted_dir, ignore_errors=True)

            if success:
                success_count += 1
            else:
                fail_count += 1
```

Also update the supported-formats print (line ~1145): change
`print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")` to
`print(f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))}")`.

- [ ] **Step 5: Run tests**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_postprocess.py -v && .venv/bin/python -c "import run_sam_interactive"`
Expected: 9 passed; import OK.

- [ ] **Step 6: Commit**

```bash
cd /home/longboardfella/sam-audio
git add run_sam_interactive.py tests/test_postprocess.py
git commit -m "feat: accept mp4/mkv inputs in CLI via ffmpeg audio extraction"
```

---

### Task 4: Handler video intake + Streamlit UI + colab flag

**Files:**
- Modify: `worker/handlers/sam_audio_cleanup.py` (import + intake ~line 273)
- Modify: `streamlit_app.py` (uploader line 201; controls ~line 184; payload ~line 215)
- Modify: `colab_smoke_test.py` (args ~line 33; payload ~line 65)

- [ ] **Step 1: Handler intake**

In `worker/handlers/sam_audio_cleanup.py`, extend the existing import from `run_sam_interactive` (lines 20-27) to also import `VIDEO_EXTENSIONS` and `extract_audio_to_wav`.

Find (~line 273):

```python
    process_input = input_path
    try:
        if trial_seconds > 0:
```

Replace with:

```python
    process_input = input_path
    try:
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            _safe_progress(progress_cb, 8, "Extracting audio track from video", "preprocess")
            process_input = extract_audio_to_wav(input_path, ffmpeg_bin, out_dir=work_dir)

        if trial_seconds > 0:
```

(`work_dir` is created just above and cleaned up by the handler's existing lifecycle, so no extra cleanup is needed; the extracted wav keeps the original stem so output naming is unchanged.)

- [ ] **Step 2: Streamlit UI**

In `streamlit_app.py`:

1. Uploader (line 201-204): change the `type` list to
   `type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "mp4", "mkv"]` and the label to `"Upload audio or video file"`.
2. After the `normalize_percent` control (~line 184), add:

```python
        loudness_normalize = st.checkbox(
            "Loudness-normalize target (-16 LUFS, -3 dB true peak)",
            value=False,
            help="Two-pass EBU R128 on target.wav only, applied after peak normalize (overrides it for target). residual.wav is untouched.",
        )
```

3. In the payload dict (lines 215-228), after `"normalize_percent": float(normalize_percent),` add:

```python
            "loudness_normalize": bool(loudness_normalize),
```

- [ ] **Step 3: colab_smoke_test flag**

In `colab_smoke_test.py`: after `ap.add_argument("--normalize-percent", ...)` add

```python
    ap.add_argument("--loudness-normalize", action="store_true")
```

and in the payload dict after `"normalize_percent": float(args.normalize_percent),` add

```python
        "loudness_normalize": bool(args.loudness_normalize),
```

- [ ] **Step 4: Verify imports and fast tests**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -c "import run_sam_interactive; import worker.handlers.sam_audio_cleanup; import colab_smoke_test" && .venv/bin/python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('streamlit OK')" && .venv/bin/python -m pytest tests/ -v -k "not Integration"`
Expected: imports OK; 20 passed (11 + 9 new), 5 deselected.

- [ ] **Step 5: Commit**

```bash
cd /home/longboardfella/sam-audio
git add worker/handlers/sam_audio_cleanup.py streamlit_app.py colab_smoke_test.py
git commit -m "feat: video input intake in worker, loudness checkbox in Streamlit, smoke-test flag"
```

---

### Task 5: End-to-end GPU verification (video in → loudnormed target out)

**Files:** none committed (verification only)

- [ ] **Step 1: Build a real test mp4 from the known-good input**

```bash
cd /home/longboardfella/sam-audio
ffmpeg -y -i /mnt/f/hf-home/audio_input/Apollo13.wav -f lavfi -i "color=c=black:s=128x72:d=58" \
  -shortest -c:v libx264 -c:a aac /tmp/apollo13_test.mp4
```

- [ ] **Step 2: Run the pipeline on it with loudness normalize enabled**

```bash
cd /home/longboardfella/sam-audio
nohup .venv/bin/python colab_smoke_test.py \
  --input /tmp/apollo13_test.mp4 \
  --model-dir ~/models/sam-audio-large-tv \
  --description "men's voices" \
  --chunk-duration 30 --overlap 2 \
  --rerank 1 --predict-spans \
  --memory-fraction 0.85 --device cuda \
  --loudness-normalize \
  --output-dir ./audio_output/loudnorm_e2e \
  > /tmp/sam_loudnorm_e2e.log 2>&1 &
```

Poll `tail -20 /tmp/sam_loudnorm_e2e.log` until done (~1-5 min warm). Any traceback → report it; do not patch code mid-verification.

- [ ] **Step 3: Verify the output**

```bash
cd /home/longboardfella/sam-audio
.venv/bin/python - <<'EOF'
import glob, zipfile, tempfile, subprocess
from pathlib import Path
from worker.handlers.sam_audio_cleanup import _measure_loudness

zips = glob.glob("audio_output/loudnorm_e2e/**/*.zip", recursive=True)
assert zips, "no result zip"
td = tempfile.mkdtemp()
with zipfile.ZipFile(zips[0]) as z:
    z.extractall(td)
target = Path(td) / "target.wav"
residual = Path(td) / "residual.wav"
mt = _measure_loudness(target, "ffmpeg")
mr = _measure_loudness(residual, "ffmpeg")
print("target  I=%s LUFS, TP=%s dB" % (mt["input_i"], mt["input_tp"]))
print("residual I=%s LUFS, TP=%s dB" % (mr["input_i"], mr["input_tp"]))
assert abs(float(mt["input_i"]) - (-16.0)) <= 1.0, "target not at -16 LUFS"
assert float(mt["input_tp"]) <= -2.9, "target true peak above -3 dB"
EOF
```

Expected: target ≈ −16 LUFS with TP ≤ −3 dB; residual unnormalized (whatever the model produced). Report both numbers.

---

### Task 6: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Features list**

In the `## Features` block, add two bullets:

```markdown
- **Video input** - MP4/MKV accepted everywhere; audio is extracted with ffmpeg before processing
- **Loudness normalization** - optional two-pass EBU R128 on `target.wav` (-16 LUFS integrated, -3 dB true peak), via Streamlit checkbox / `loudness_normalize` payload / `--loudness-normalize`
```

- [ ] **Step 2: Commit**

```bash
cd /home/longboardfella/sam-audio
git add README.md
git commit -m "docs: document video input and loudness normalization"
```

---

## Self-review notes

- Spec coverage: parser+tests (T1), two-pass loudnorm + payload + target-only post-process ordering (T2), extraction helper + CLI + dir scan (T3), handler intake + Streamlit + colab flag (T4), round-trip and e2e verification incl. ±1 LU / TP assertions (T2/T5), README (T6). CLI loudness prompt deliberately absent (spec non-goal).
- Working tree may contain unrelated untracked files (screenshots, zip): every commit step lists explicit paths; never `git add -A`.
- Type consistency: `_parse_loudnorm_json(str) -> dict`, `_measure_loudness(Path, str) -> dict`, `_apply_loudness_normalize(Path, str) -> None`, `extract_audio_to_wav(Path, str, Optional[Path]) -> Path` used consistently across tasks.
