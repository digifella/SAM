# Loudness Normalization + Video Input — Design

**Date:** 2026-06-11
**Status:** Approved

## Problem

1. The only output normalization is linear peak-normalize-to-percent,
   applied to both target and residual. The user wants a one-click
   "broadcast-style" option: consistent average loudness on the cleaned
   target with a hard peak ceiling.
2. Inputs are limited to audio containers (`wav/mp3/flac/ogg/m4a/aac`).
   Source material often arrives as video (`.mp4`, `.mkv`); today the
   audio must be extracted manually first.

## Goals

- Checkbox in Streamlit (and a flag on `colab_smoke_test.py`):
  **loudness-normalize `target.wav` to −16 LUFS integrated with a −3 dB
  true-peak ceiling** (EBU R128 via ffmpeg `loudnorm`). Default off.
  Applies to `target.wav` only — `residual.wav` is untouched.
- Accept `.mp4` and `.mkv` inputs in all three frontends (interactive
  CLI directory scan, Streamlit uploader, queue worker payload), with a
  single ffmpeg audio-extraction step before the normal pipeline.

## Non-Goals

- No RMS or single-pass dynamic normalization (rejected: single-pass
  loudnorm applies dynamic gain that can pump on speech).
- No pyloudnorm dependency (ffmpeg already does this).
- No loudness option for residual output.
- No other video containers (`.mov`, `.webm`, …) until asked. YAGNI.
- No loudness prompt in the interactive CLI: post-processing lives in
  the worker handler, which the CLI bypasses. CLI users who want
  loudnorm use the Streamlit harness or `colab_smoke_test.py` locally.
  (Video extraction, by contrast, IS wired into the CLI — it happens at
  intake, not post-process.)

## Design

### 1. Loudness normalization (worker handler)

New helper in `worker/handlers/sam_audio_cleanup.py`:

- `_measure_loudness(path, ffmpeg_bin) -> dict` — runs
  `ffmpeg -i path -af loudnorm=I=-16:TP=-3:LRA=11:print_format=json -f null -`,
  parses the JSON block from stderr (keys: `input_i`, `input_tp`,
  `input_lra`, `input_thresh`, `target_offset`). The JSON parsing lives
  in its own pure function `_parse_loudnorm_json(stderr_text) -> dict`
  so it is unit-testable without ffmpeg.
- `_apply_loudness_normalize(path, ffmpeg_bin) -> None` — second pass:
  `ffmpeg -i path -af loudnorm=I=-16:TP=-3:LRA=11:measured_I=…:measured_TP=…:measured_LRA=…:measured_thresh=…:offset=…:linear=true -ar <orig_rate> out.wav`,
  then atomically replaces the original. `linear=true` ensures a single
  clean gain (no dynamics processing). Output sample rate pinned to the
  input's rate (loudnorm upsamples to 192k internally; we resample back).

Payload key: `loudness_normalize` (bool, default false), parsed with the
existing `_as_bool`. Ordering in the post-process stage: trial-trim →
peak-normalize (existing, both files) → **loudness-normalize (target
only)** → transcode. If both normalize options are set, loudnorm runs
last and determines the final target level.

Errors: a non-zero ffmpeg exit or unparseable JSON raises RuntimeError
with ffmpeg's stderr tail — the job fails visibly, never silently skips.

### 2. Video-container input

- `run_sam_interactive.py`: add `VIDEO_EXTENSIONS = {'.mp4', '.mkv'}`
  next to `AUDIO_EXTENSIONS`; the directory scan globs both sets. New
  helper `extract_audio_to_wav(path, ffmpeg_bin="ffmpeg") -> Path`:
  `ffmpeg -y -i in -vn -acodec pcm_s16le -ar 48000 -ac 2 out.wav` to a
  temp file; raises RuntimeError (with stderr tail) if ffmpeg fails,
  e.g. no audio stream. The CLI calls it for video inputs before
  `process_audio_file`; the extracted temp wav is deleted after the file
  completes (same pattern as the existing mono16k temp handling).
- `worker/handlers/sam_audio_cleanup.py`: at intake, if the input
  suffix is in `VIDEO_EXTENSIONS` (imported from `run_sam_interactive`),
  call `extract_audio_to_wav` into the job work dir and continue the
  pipeline on the extracted wav. Output naming still derives from the
  original stem.
- `streamlit_app.py`: uploader `type` list gains `"mp4"`, `"mkv"`; new
  checkbox "Loudness-normalize target (−16 LUFS, −3 dB true peak)"
  below the existing normalize control, wired into the payload; caption
  notes it overrides the peak-% control for `target.wav`.
- `colab_smoke_test.py`: `--loudness-normalize` store-true flag →
  payload.

### 3. Testing

- Unit (no ffmpeg): `_parse_loudnorm_json` on a captured stderr sample
  (valid, missing-JSON, malformed-JSON cases); video-extension routing.
- Round-trip (ffmpeg, no GPU): generate a 2 s tone mp4 with ffmpeg
  (`-f lavfi -i sine`, `-f lavfi -i color` for a video track), run
  `extract_audio_to_wav`, assert duration/sample rate; run the two-pass
  loudnorm on a quiet generated wav and assert measured integrated
  loudness lands within ±1 LU of −16 and peak ≤ −3 dBFS.
- Existing suite stays green.

## Risks

- **loudnorm JSON format drift across ffmpeg versions** — parsing is
  isolated in one pure function with tests; failure mode is a visible
  job error.
- **Very short clips (<3 s)**: integrated loudness needs ~0.4 s minimum
  and is unreliable under ~3 s; loudnorm still produces output (it
  falls back to dynamic mode internally if linear can't hit target —
  acceptable, and rare for this use case).
- **Video with multiple audio tracks**: ffmpeg picks the default/first
  track; documented behaviour, no track-selection UI.
