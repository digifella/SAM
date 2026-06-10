# SAM-Audio Queue Worker

This worker connects `sam-audio` to the website queue (`queue_worker_api.php`) and handles `sam_audio_cleanup` jobs.

## Setup

1. Create config:
```bash
cp worker/config.env.example worker/config.env
```

2. Edit `worker/config.env` with your queue URL and secret key.

3. Install dependencies in your environment:
```bash
pip install requests soundfile numpy torch
```

4. Ensure `ffmpeg` is installed and available in `PATH`.

## Run

```bash
python worker/worker.py
```

Optional custom config path:
```bash
python worker/worker.py --config /path/to/config.env
```

## Job Contract

Expected queue type: `sam_audio_cleanup`

Input file: audio upload from website (`wav/mp3/flac/ogg/m4a/aac`).

`input_data` options supported now:
- `description`: text prompt to extract (default `speech`)
- `chunk_duration`: seconds per chunk (default `60`)
- `overlap`: chunk overlap seconds (default `2.0`)
- `convert_to_mono`: pre-convert to mono 16k before separation (default `true`)
- `rerank`: rerank candidates 1-8 (default `1`)
- `predict_spans`: enable SAM span prediction (default `false`)
- `trial_seconds`: process only first N seconds (optional)
- `normalize_percent`: peak-normalize output to 0-100% full-scale (optional)
- `output_sample_rate`: transcode output sample rate, e.g. `32000` (optional)
- `output_channels`: `1` mono or `2` stereo (optional)
- `memory_fraction`: CUDA memory fraction (default `0.85`)
- `model_dir`: model path override (optional)
- `device`: `auto`, `cuda`, or `cpu`

Output file: ZIP uploaded to queue completion with:
- `target.wav`
- `residual.wav`
- `metadata.json`

`output_data` includes duration, chunk count, sample rate/channels, and applied options.

## Power Window

The worker does not enforce time windows locally. Your website queue controls this in `poll` (already configured for 11:00-14:00 schedule rules).
