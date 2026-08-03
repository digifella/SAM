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

For audio files longer than the chunk duration (default 30s):
- Files are split into overlapping chunks
- Each chunk is processed independently using streaming (one at a time)
- Results are merged with crossfade at overlap regions
- Memory usage stays constant regardless of file length

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
