# Repository Guidelines

## Project Structure & Module Organization
Core Python entry points (`run_sam_audio.py`, `run_sam_audio_batch*.py`, `streamlit_app.py`) live in the repo root for quick CLI or UI runs. Shared helpers sit in `sam_audio_utils/` (`audio.py` for ffmpeg conversion + tensor utilities, `paths.py` for env-aware locations). Interactive assets and cached audio land in `audio_input/`, `audio_output/`, `audio_cache/`, and `audio_work/`. Keep large model checkpoints outside the repo (`$SAM_MODEL_DIR`, defaults to `~/models/sam-audio-large-tv`) and reference them via env vars.

## Build, Test, and Development Commands
Use Python 3.10+ with CUDA-enabled PyTorch when available. Typical workflow:
- `python -m venv .venv && source .venv/bin/activate` then `pip install -r streamlit_requirements.txt` to get Streamlit + SAM deps.
- `streamlit run streamlit_app.py` launches the UI; pass `SAM_OUTPUT_DIR=/tmp/sam-out streamlit run ...` to adjust storage.
- `python run_sam_audio.py --file audio_input/example.wav --desc "female narration"` runs a single inference and writes `*_target.wav` and `*_residual.wav`.
- `python run_sam_audio_batch.py --pattern "*.wav" --auto_convert_16k` iterates a directory, applying chunked fallback when long files are detected.
- `docker build -f Dockerfile.streamlit -t sam-audio-ui .` followed by `docker run -p 8501:8501 sam-audio-ui` gives a reproducible UI environment.

## Coding Style & Naming Conventions
Stick to 4-space indentation, type hints, and `pathlib.Path` for filesystem logic. Favor f-strings, explicit env-variable plumbing (`SAM_*` names), and pure functions in `sam_audio_utils/`. Keep tensors as `torch.float32` until final conversion, and gate device-specific logic (CUDA) behind `torch.cuda.is_available()`. Configuration flags should mirror existing kebab-case CLI options and uppercase env vars.

## Testing Guidelines
There is no automated suite yet; add targeted `pytest` modules under `tests/` when extending core logic. Mock GPU availability and ffmpeg binaries so unit tests stay CPU-bound. For manual smoke tests, run `python run_sam_audio.py ...` on a 10 s clip and `streamlit run ...` to confirm upload, conversion, and download flows. Document new sample assets (duration, source) inside the PR to preserve reproducibility.

## Commit & Pull Request Guidelines
This repo is frequently vendored outside Git, so keep commits atomic with imperative subjects (`feat: add chunked batch guard`, `fix: clamp rerank fallback`). Reference relevant scripts or env vars in the body and mention manual test coverage (`Test: python run_sam_audio.py ...`). PRs should include: summary of user impact, configuration/env var changes, screenshots or CLI logs for UI-facing updates, and any new data requirements. Flag GPU memory considerations and ffmpeg dependencies so reviewers can reproduce your results.

## Environment & Security Notes
Never check in model weights or licensed audio. Use env vars (`SAM_OUTPUT_DIR`, `SAM_CACHE_DIR`, `SAM_FFMPEG_BIN`, `SAM_GPU_MEM_FRAC`, `SAM_OFFLINE`) instead of hardcoding paths, and scrub temporary WAVs from shared drives when processing sensitive material. When running offline, set `SAM_OFFLINE=1` (or pass `--offline` in batch scripts) to avoid accidental Hugging Face traffic.
