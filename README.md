# SAM-Audio Interactive Processor

Memory-optimized interactive batch processor for SAM-Audio with chunking support and WSL2 stability improvements.

## Features

- **Memory-safe streaming processing** - Processes audio chunks one at a time to prevent memory exhaustion
- **Interactive configuration** - Save and reuse your processing settings
- **Automatic chunking** - Handles long audio files by splitting into manageable chunks with overlap
- **WSL2 optimized** - Aggressive memory cleanup prevents crashes on WSL2 systems
- **Progress tracking** - Real-time progress updates and detailed logging
- **GPU accelerated** - CUDA support with configurable memory limits
- **Video input** - MP4/MKV/MOV accepted everywhere; audio is extracted with ffmpeg before processing
- **Automatic input pre-gain** - input peak is measured (ffmpeg volumedetect) and quiet inputs are boosted toward -3 dBFS (max +30 dB, never boosts silence); hot inputs are pulled back. No user setting needed
- **Loudness normalization** - optional two-pass EBU R128 on `target.wav` (-16 LUFS integrated, -3 dB true peak), via Streamlit checkbox / `loudness_normalize` payload / `--loudness-normalize`

## Memory-Optimized Local Loading (RTX 8000 / 48GB)

`sam_audio_local/loader.py` loads SAM-Audio large in ~13GB resident VRAM
(vs ~31GB stock) by:

- stripping the vision encoder and ImageBind/CLAP/Judge rerankers
  (text-prompted separation only; `rerank` is pinned to 1)
- casting the DiT/codec to fp16 (native on Turing); the T5 text encoder
  and span predictor stay fp32
- mmap-loading the 14.9GB checkpoint so model load stays under the WSL2
  system-RAM ceiling

All entry points (`run_sam_interactive.py`, `streamlit_app.py`, the queue
worker) use this loader automatically. Video prompting and reranking
require the stock `SAMAudio.from_pretrained()` path (e.g. on Colab).

## Quick Start

```bash
# Run the interactive processor
python run_sam_interactive.py
```

## Queue Worker (Website Integration)

This repo now includes a queue worker compatible with your website queue API (`/admin/queue_worker_api.php`), matching the same pattern used in `cortex_suite`.

```bash
cp worker/config.env.example worker/config.env
python worker/worker.py
```

Details and supported queue parameters are documented in `worker/README.md`.

## Local Streamlit Harness

Use the local GUI to test the same processing path as queue jobs (`sam_audio_cleanup` handler):

```bash
streamlit run streamlit_app.py
```

The app lets you upload one file, tune key options (trial seconds, normalize %, sample rate/channels, chunking, rerank), run local processing, preview `target.wav` and `residual.wav`, and download the ZIP output.

The local Streamlit upload limit is set in `.streamlit/config.toml` with `server.maxUploadSize = 4096`, allowing files up to 4 GB.

## `clean_cli.py` (Headless / Bridge Integration)

A CLI wrapper around the same `sam_audio_cleanup` handler used by the Streamlit
harness and the queue worker, intended for subprocess use — e.g. a Streamlit
page spawning a foreground job, or the kb-query-server bridge detached-spawning
a job for a Discord request. Progress streams as JSON lines on stdout; the
final outcome lands in `<out-dir>/status.json`.

```bash
python clean_cli.py --input <audio_or_video_file> \
    --job-json <path_to_job.json> \
    --out-dir <output_dir> \
    [--opus] [--notify-discord] [--gpu-lock-timeout <seconds>]
```

### Flags

- `--input` (required) — path to the source audio/video file.
- `--job-json` (required) — path to a JSON file with the job payload (same
  fields as the Streamlit/worker payload: `description`, `convert_to_mono`,
  `chunk_duration`, `overlap`, `rerank`, `predict_spans`, `normalize_percent`,
  `output_sample_rate`, `output_channels`, `memory_fraction`, etc.).
- `--out-dir` (required) — directory for `status.json`, `result.zip`, and
  (with `--opus`) `target.ogg`. Created if missing.
- `--opus` — after separation, extract `target.wav` from the result ZIP and
  encode it to mono Opus (`target.ogg`) using a bitrate retry ladder
  (`48k` → `24k`) to try to land under Discord's 25MB upload cap.
- `--notify-discord` — POST a summary (and the Opus file, if produced) to a
  Discord webhook on completion or failure. The webhook URL is **never** a
  CLI argument — it is read from the `DISCORD_WEBHOOK_URL` environment
  variable only. If the flag is set but the env var is unset/empty, the CLI
  emits a `notify` progress line saying so and continues without notifying
  (this is not a job failure).
- `--gpu-lock-timeout` — seconds to wait for the exclusive GPU lock
  (`~/.sam_audio_gpu.lock`) before giving up (default `7200`). The lock
  serializes concurrent CLI invocations so only one job touches the GPU at a
  time; while waiting, a `queue` progress line is emitted every ~30s.

### Stdout protocol

Every progress update is a single JSON object per line:

```json
{"type": "progress", "pct": 35, "stage": "separate", "message": "Running SAM-Audio separation (attempt 1/3: requested)", "ts": "14:32:39"}
```

`pct` is 0-100, `stage` is a short machine-readable phase name (`prepare`,
`model`, `separate`, `postprocess`, `package`, `encode`, `notify`, `queue`,
`complete`, ...), and `ts` is a `HH:MM:SS` timestamp. Consumers should parse
stdout line-by-line and ignore any non-JSON lines (model/library warnings can
still land on stdout/stderr around it).

### `status.json` states

- `"running"` — written immediately on start, with `pid`, `started_at`, and
  `input`. A bridge that finds `status.json` in this state treats the job as
  still in progress.
- `"done"` — separation succeeded. Includes `metadata` (the handler's
  `output_data`: duration, chunks processed, sample rate/channels, the
  options actually applied, and an optional `warning` string) and
  `finished_at`. **A `"done"` state does not guarantee a good result** — the
  handler can flag a near-silent target (the description didn't match any
  sound in the audio) via `metadata.warning` while still reporting `"done"`.
  Callers that care about audio quality should check for `metadata.warning`,
  not just the state.
- `"error"` — an exception was raised (bad job-json, handler failure, GPU
  lock timeout, etc.). Includes `error` (stringified exception) and
  `finished_at`.

### `--opus` failure handling

If Opus encoding fails (e.g. `ffmpeg` missing or crashes) **after** separation
has already succeeded and `result.zip` is valid on disk, the CLI treats this
as a delivery problem, not a job failure: it keeps `state: "done"`, appends an
`opus encode failed: ...` note to `metadata.warning` (preserving any prior
warning), and — if `--notify-discord` is set — still posts the text summary,
just without the audio attachment. The job only becomes `state: "error"` if
separation itself fails.

### Result contents

`result.zip` contains `target.wav`, `residual.wav`, and `metadata.json`. With
`--opus`, `target.ogg` (mono Opus) is written alongside `status.json` and
`result.zip` in `--out-dir`.

## Cortex Suite page

`cortex_suite/pages/21_Audio_Cleanup.py` (Cortex v6.7.0) exposes this pipeline in
the Cortex Suite UI. It spawns `clean_cli.py` in **this** project's venv as a
subprocess, so no torch or SAM dependency enters the cortex venv. Progress arrives
as JSON lines on the subprocess stdout; the result downloads as a ZIP and both
stems play inline.

`SAM_AUDIO_ROOT` overrides this repo's location (default `/home/longboardfella/sam-audio`);
the page shows a clear error if that venv is missing rather than failing at spawn.

## Discord loop (Hermes)

Paul attaches a wav/mp3 in Discord and asks Hermes to clean it up. **Discord is only
the transport** — this is not for cleaning Discord's own voice messages.

```
Discord attachment
   -> Hermes on sp4 runs  ~/.local/bin/kb-audio-clean <file> "<sound to KEEP>"
   -> POST /audio-clean on the fastfella bridge (100.118.92.17:7333, bearer token)
   -> bridge spools the upload and spawns clean_cli.py DETACHED, returns a job id
   -> clean_cli separates on the GPU, encodes Opus, and POSTs the result
      to the channel webhook itself
```

The bridge never blocks on the job; the caller may poll `GET /audio-status?id=<job_id>`.

- **Spool:** `~/sam-audio/spool/<job_id>/` — `input.<ext>`, `job.json`, `job.log`,
  `status.json`, `result.zip`, `target.ogg`. Purged after 7 days at bridge start.
- **Webhook:** `~/.discord-audio-webhook` (chmod 600), read at spawn and passed as
  `DISCORD_WEBHOOK_URL`. Never on argv, so it stays out of `/proc/*/cmdline`.
- **Bridge endpoints** are documented in
  `nemoclaw_ops/docs/2026-08-03-kb-bridge-audio-clean-endpoints.md`. The bridge
  script itself is not in git; a backup lives at `~/kb-query-server.py.bak-20260803`.

**The description here was historically framed as what to EXTRACT, not what to
remove**, and the `kb-audio-clean` argument above is still written as
`"<sound to KEEP>"`. That framing is no longer the whole picture — `clean_cli.py`
itself now also supports removal phrasing directly; see "Removing a sound vs.
extracting one" below for the current contract. Whether removal phrasing sent to
Hermes survives intact through to `clean_cli.py`, or is still translated to a
"sound to keep" phrase before it arrives, is **unverified from this repo** — that
translation lives on sp4, outside this codebase, and its documented default is
`"a person speaking"`.

Discord's attachment cap (25MB for non-Nitro) bounds the input: roughly 2 minutes of
48kHz mono WAV, or ~25 minutes of 128kbps MP3. Results come back as Opus with a
bitrate ladder to stay under the same cap.

## Denoising (default cleanup path)

`clean_cli.py` denoises by default and only routes to SAM-Audio separation on positive
evidence of a mixture. SAM-Audio is source separation, not noise reduction — on the real
9m39s single-speaker test file (`Fionnuala_raw.wav`) SAM returned a "cleaned" target that
was 93.5% sub-300Hz rumble with 0.1% speech-band energy (the voice went to the residual
instead), and the pipeline reported success. The denoiser exists to fix that.

The routing rule follows from an asymmetry: an under-cleaned file is still listenable,
whereas SAM on single-speaker noise returns an essentially empty stem. Ambiguity
therefore resolves to denoise. `choose_method()` only routes to `separate` when the
description names a source to extract *and* implies a mixture — a named second source
(`_MIXTURE_WORDS`: guitar, dog, siren, music, ...) or a phrase implying a competing sound
(`_MIXTURE_PHRASES`: " over ", " behind ", " through ", " on top of ", " against "). Everything
else — "clean this up", "remove background noise", "a person speaking", an empty
description — denoises.

### Removing a sound vs. extracting one

Both are supported, and the difference is what lands in `target.wav`.

- **Name a sound** — `"the guitar"` — and SAM extracts it: `target.wav` is the guitar.
- **Ask for a sound to go** — `"remove the guitar"` — and you get the opposite:
  `target.wav` is everything *except* the guitar, and `residual.wav` holds the guitar
  that was taken out.

Removal only reaches SAM when the sound you name is one it can extract. "Remove the
background noise" and "remove the hiss" still denoise, because hiss is not a source —
asking SAM to extract it returns a near-empty stem.

If SAM cannot find the sound you asked to remove, the result is approximately your
original audio: nothing was removed, but nothing was lost either.

### Known routing limitations

Routing is keyword matching over a free-text description, so it has edges. Set `method`
explicitly in `job.json` whenever you know which one you want — it always beats inference.

- **A mixture word that is the topic of the speech misroutes to `separate`.** "A lecture
  about classical music history" and "clean up this interview about traffic policy" both
  name a source that is being *talked about*, not heard. Distinguishing the two needs more
  than keyword matching.
- **Mixture words match whole words plus a plural `s` only.** "Two guitars" counts;
  a bare "barking outside" or "drumming in the next room" does not, and will denoise.
  This is deliberate — matching `-ed`/`-ing` would read "alarm" out of "alarming" and
  "crowd" out of "crowded", sending ordinary cleanup requests to SAM.
- **Device words need a playing cue.** "Radio", "phone", "tv" and "television" name the
  recording medium as often as a second source, so they only imply a mixture alongside a
  cue that the device is audibly playing ("a tv **playing** in the background"). "Clean up
  this phone call recording" denoises.
- **A removal request whose subject is the topic of the speech misroutes.** "Remove the
  part about traffic" names a subject being discussed, not a sound, and will route to
  `remove`. The two-sided guard should catch the resulting voice loss and warn, but set
  `method` explicitly when you know.

### `job.json` keys

No new CLI flags were added — the bridge, Cortex and the sp4 wrapper send exactly what
they already send, and routing reads two keys already carried in the existing payload:

```json
{"description": "...", "method": "auto|denoise|separate|remove", "strength": "gentle|normal|aggressive"}
```

- `method` — `"auto"` (default), `"denoise"`, `"separate"`, or `"remove"`. An explicit
  value always beats inference. `"remove"` runs SAM on the named sound and returns the
  residual as `target.wav`.
- `strength` — one of `STRENGTH_PRESETS`: `"gentle"`, `"normal"` (default), or
  `"aggressive"`, trading off how much noise is removed against the risk of musical-noise
  warble.

### No GPU, no lock

`sam_audio_utils/denoise.py` is pure numpy/scipy spectral subtraction — no torch, no
model, no GPU. `clean_cli.py` skips `gpu_lock` entirely on this path, so a denoise job
runs immediately even while a SAM job holds the card.

### The silence gate

`Fionnuala_raw.wav` is a collation of separate cuts, so roughly a quarter of it is
absolute digital zero — the quietest thing in the file. A noise profile estimated from
those frames would be all zeros: it subtracts nothing while appearing to succeed, which
is the same shape of failure as the SAM incident. Frames at or below `SILENCE_GATE_DBFS`
(-90 dBFS) are therefore excluded from noise profiling and from the quietest-window
search, and any run of true digital silence in the input is written back out as exact
zero rather than left to be smeared by STFT overlap-add.

### Checking nothing was eaten

`residual.wav` (`= input - target`) is the diagnostic: listen to it, and if you can hear
the speaker, the settings were too aggressive. That is exactly how the original SAM
failure was found — the residual had the voice in it, not the target.

On the real test file the automated result correlates **0.9983** with Paul's hand-tuned
WavePad output (the oracle), removes **19.3 dB** of noise floor versus WavePad's 18.8 dB,
and retains **0.9051** of speech-band energy versus WavePad's 0.9078 — against SAM's
0.0018 on the same file. Sample rate is preserved end to end: 44.1kHz in, 44.1kHz out,
no resampling.

## Colab Smoke Test (No Streamlit)

To test the same processing pipeline on Google Colab (single file, conservative memory settings), use:

- Notebook: `colab/SAM_Audio_Colab_Smoke_Test.ipynb`
- CLI entrypoint: `colab_smoke_test.py`

The notebook installs dependencies, uploads one audio file, downloads model weights, runs `sam_audio_cleanup`, and downloads the result ZIP.

The script will prompt you for:
- Input directory containing audio files
- Text description of audio to extract
- Output directory for results
- Model directory path
- Processing parameters (chunk size, overlap, GPU memory, etc.)

Settings are saved to `~/.sam_audio_config.json` for future runs.

## Requirements

- Python 3.11+
- CUDA-capable GPU (tested with 48GB VRAM)
- SAM-Audio model files
- Dependencies: torch, soundfile, numpy, etc.

## Processing Details

### Chunking

Files are split into overlapping chunks:
- Each chunk is processed independently using streaming (one at a time)
- Results are merged with crossfade at overlap regions
- Memory usage stays constant regardless of file length

**Short files are chunked too, deliberately.** A file shorter than
`chunk_duration` used to take a single whole-file inference path, and that path
is unreliable — measured silent (target below -70 dBFS) in **4 of 6 runs**, while
the chunked path was healthy in 4 of 4. Files at or under `chunk_duration` now
force chunking by roughly halving the effective chunk size
(`FORCE_CHUNK_MIN_SECONDS = 3.0`, so inputs under ~6s still fall back rather than
minting an unusably small chunk). Long-file behaviour is unchanged — the
validated 45-minute soak profile produces byte-identical boundaries.

A trailing chunk shorter than `MIN_CHUNK_SECONDS` (0.5s) is folded into the
previous chunk rather than emitted: the codec's reflect-pad requires input longer
than `hop_length - 1` (1919 samples, ~40ms), and a sub-second remainder crashed
with *"Padding size should be less than the corresponding input dimension"*. The
fold never collapses a file to a single chunk — if that would happen, the file is
re-split evenly instead, since one whole-file chunk is exactly the unreliable
path above.

### Memory Management

The script uses several techniques to prevent memory exhaustion:
1. **Streaming generator** - Loads one audio chunk at a time
2. **Aggressive cleanup** - Deletes tensors and runs garbage collection after each chunk
3. **CPU offloading** - Moves tensors to CPU before reranking to free GPU memory
4. **Configurable limits** - Set maximum files per session and GPU memory fraction

### Output

For each input file, two output files are created:
- `{filename}_target.wav` - Extracted audio matching the description
- `{filename}_residual.wav` - Remaining audio (background/noise)

## Configuration

Default settings (stored in `~/.sam_audio_config.json`):
```json
{
  "input_dir": "/path/to/input",
  "output_dir": "/path/to/output",
  "description": "softly spoken woman talking",
  "model_dir": "/path/to/model",
  "rerank": 1,
  "predict_spans": false,
  "chunk_duration": 30,
  "overlap": 2.0,
  "memory_fraction": 0.85,
  "convert_to_mono": true
}
```

## Logs

Processing logs are saved to `~/.sam_audio_logs/` with detailed memory statistics for debugging.

## Troubleshooting

See `CRASH_FIX_2026-01-12.md` for detailed information about:
- Memory exhaustion fixes
- Infinite loop bug resolution
- WSL2 stability improvements
- Performance optimization details

## Development History

Previous versions and development artifacts are archived in the `archive/` directory. See `archive/ARCHIVE_INDEX.md` for details.

## License

This is a wrapper/utility script for SAM-Audio. See the original SAM-Audio project for model licensing.

## Credits

Built to solve memory exhaustion issues when processing long audio files with SAM-Audio on WSL2 systems.
