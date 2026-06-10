# SAM-Audio RTX 8000 Memory-Optimized Loader — Design

**Date:** 2026-06-11
**Status:** Approved (Option A)

## Problem

Local runs on the Quadro RTX 8000 (48GB VRAM) OOM regardless of chunking
parameters, forcing all processing onto Colab. Two compounding causes:

1. **VRAM:** `SAMAudio.from_pretrained()` loads everything in fp32:
   - DiT transformer: 11.72 GB (from checkpoint)
   - Vision encoder (PE-Core-L14-336): 2.68 GB (from checkpoint)
   - Audio codec (DACVAE): 0.43 GB (from checkpoint)
   - T5-base text encoder, ImageBind visual ranker, CLAP + Judge text
     rankers, pe-a-frame-large span predictor: downloaded from HF at
     `__init__`, **not** in the checkpoint — roughly another 15 GB fp32.

   Measured resident footprint: **30.8 GB** before any audio is processed
   (CRASH_FIX_2026-01-12.md). ODE-solver activations on 30 s / 48 kHz
   chunks exhaust the remaining ~17 GB.

2. **System RAM:** `BaseModel._from_pretrained` instantiates the full
   fp32 model on CPU, then `torch.load`s the entire 14.9 GB checkpoint
   without `mmap`. Peak CPU RAM ≈ model + checkpoint ≈ 30 GB, above the
   23 GB WSL2 limit → OOM-killer.

## Goals

- Run SAM-Audio large end-to-end on the RTX 8000 locally with comfortable
  VRAM headroom (~7–8 GB resident weights, ~40 GB free for activations).
- Keep: text-prompted extract/remove, span prediction
  (`predict_spans: true`), existing chunking/merge logic, interactive
  script, Streamlit harness, queue worker.
- Stay under the 23 GB WSL2 RAM ceiling during model load.

## Non-Goals

- Video/visual prompting (vision encoder + ImageBind visual ranker are
  stripped).
- Reranking (CLAP + Judge stripped; `rerank` is forced to 1). Saved
  config already uses `rerank: 1`, where rankers are dead weight.
- Forking or vendoring the upstream `sam_audio` package (Option B) or
  adopting the `sam-audio-local` CLI (Option C) — rejected in favour of a
  loader that lives in this repo.

## Design (Option A — repo-local memory-optimized loader)

### New module: `sam_audio_local/loader.py`

A single entry point, e.g. `load_sam_audio_optimized(model_dir, device)`:

1. **Strip rankers via config:** read `config.json`, set
   `visual_ranker = None` and `text_ranker = None`.
   `create_ranker(None)` already returns `None` upstream, so no patching
   is needed for these.
2. **Stub the vision encoder:** `SAMAudio.__init__` constructs
   `PerceptionEncoder(cfg.vision_encoder)` unconditionally. Temporarily
   patch it (context-managed monkeypatch of
   `sam_audio.model.model.PerceptionEncoder`) with a zero-parameter stub
   that raises if ever called, so no PE weights are downloaded or
   allocated.
3. **Avoid the CPU RAM spike:** instantiate the model with empty/meta
   weights for the checkpoint-backed modules, load the checkpoint with
   `torch.load(..., mmap=True, weights_only=True)`, filter out
   `vision_encoder.*` keys, and `load_state_dict(strict=False,
   assign=True)`. Verify that the only missing/unexpected keys are the
   deliberately stripped modules; raise otherwise.
4. **Precision:** cast checkpoint-backed modules (DiT, DACVAE, proj
   layers) to **fp16** (native on Turing). The T5 text encoder stays
   **fp32** (T5 is fp16-fragile; ~0.5 GB). Span predictor
   (pe-a-frame-large) loads as upstream does; fp16-cast it only if
   verification shows identical span output, otherwise leave fp32.
5. **Inference:** the model is half-precision, so the memory-safe
   `separate()` patch casts inputs to each module's dtype at the
   boundaries (fp16 into codec/DiT, fp32 into T5) and returns fp32
   audio. No global autocast.

Expected resident VRAM: DiT 5.9 GB + codec 0.2 GB + T5 0.5 GB + span
predictor ~1 GB ≈ **7–8 GB** (vs 30.8 GB today).

### Call-site changes

- `run_sam_interactive.py`: replace
  `SAMAudio.from_pretrained(...).eval().to(device)` with the new loader.
  The memory-safe `separate()` monkeypatch and chunk merging stay.
  Rerank prompt: hidden or pinned to 1 with an "unavailable in local
  optimized mode" note — not silently ignored.
- `streamlit_app.py` and `worker/` handler: same loader swap; rerank
  option likewise pinned/hidden.

### Error handling

- Loader fails fast with a clear message if checkpoint keys don't match
  expectations (guards against upstream package/checkpoint version
  drift).
- If CUDA is unavailable, fall back to CPU fp32 (slow but functional) —
  matches existing behaviour.

## Verification

1. **Unit-level:** loader smoke test asserting (a) no `vision_encoder`
   parameters on the model, (b) rankers are `None`, (c) DiT dtype is
   fp16, (d) resident VRAM after load < 10 GB.
2. **End-to-end:** process one real input file from
   `/mnt/f/hf-home/audio_input` with saved settings (30 s chunks, 2 s
   overlap, "men's voices", spans on) while logging `nvidia-smi` and
   system RAM. Success: completes without OOM, produces `target.wav` /
   `residual.wav`.
3. **Quality:** listening check of outputs; compare against a known-good
   Colab output for the same input if available.

## Risks

- **fp16 numeric drift** in the ODE solve or DACVAE decode → audible
  artifacts. Mitigation: listening check; per-module fp32 fallback knobs
  if needed.
- **`assign=True` + mmap interaction**: tensors remain disk-backed until
  moved to GPU, so `.to(device)` streams the checkpoint from disk. The
  checkpoint lives on ext4 (`~/models`), not a Windows mount, so this is
  fast; the fp16 cast also materializes tensors in RAM incrementally
  (one tensor at a time), staying far below the 23 GB ceiling.
- **Upstream drift**: pinned by the fail-fast key verification in the
  loader.
