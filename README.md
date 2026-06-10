# SAM-Audio Interactive Processor

Memory-optimized interactive batch processor for SAM-Audio with chunking support and WSL2 stability improvements.

## Features

- **Memory-safe streaming processing** - Processes audio chunks one at a time to prevent memory exhaustion
- **Interactive configuration** - Save and reuse your processing settings
- **Automatic chunking** - Handles long audio files by splitting into manageable chunks with overlap
- **WSL2 optimized** - Aggressive memory cleanup prevents crashes on WSL2 systems
- **Progress tracking** - Real-time progress updates and detailed logging
- **GPU accelerated** - CUDA support with configurable memory limits

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
