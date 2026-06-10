# RTX 8000 Memory-Optimized SAM-Audio Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load SAM-Audio large in ~7–8GB VRAM (fp16, vision encoder + rankers stripped, mmap checkpoint load) so the full pipeline runs locally on the Quadro RTX 8000 without OOM.

**Architecture:** A new repo-local package `sam_audio_local/` provides `load_sam_audio_optimized()`. It strips rankers via config (upstream `SAMAudioConfig` accepts `visual_ranker=None, text_ranker=None` natively), swaps the vision encoder for a zero-parameter stub via a context-managed monkeypatch, mmap-loads the checkpoint, and casts checkpoint-backed modules to fp16. The two existing load sites (`run_sam_interactive.py:1163`, `worker/handlers/sam_audio_cleanup.py:64`) switch to it. dtype casts at module boundaries go into the existing `memory_safe_separate()` in `run_sam_interactive.py`, which the worker and Streamlit app already reuse.

**Tech Stack:** Python 3.11, torch 2.6.0+cu124, `sam_audio` 0.1.0 (pip package, untouched), pytest 9. All commands run with the project venv: `/home/longboardfella/sam-audio/.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-11-rtx8000-memory-optimized-loader-design.md`

---

## Background facts (verified against installed code — do not re-derive)

- `SAMAudioConfig` (`sam_audio/model/config.py:204`) accepts `visual_ranker=None` / `text_ranker=None`; `create_ranker(None)` returns `None`. No patching needed to strip rankers.
- `SAMAudio.__init__` (`sam_audio/model/model.py:75-104`) constructs `PerceptionEncoder(cfg.vision_encoder)` unconditionally — this is the only thing requiring a monkeypatch.
- On the text-only path (`batch.masked_video is None`), `_get_video_features` reads **only** `self.vision_encoder.dim` (`model.py:189`). The stub needs a `.dim` attribute and nothing else.
- `SAMAudio.load_state_dict(sd, strict=True)` (`model.py:346`) internally calls `super().load_state_dict(strict=False)` and tolerates missing keys matching `^text_encoder|^visual_ranker|^text_ranker|^span_predictor` (those load from HuggingFace at `__init__`, not from the checkpoint); it raises on any other missing/unexpected key. So after filtering `vision_encoder.*` keys out of the checkpoint, a plain `model.load_state_dict(filtered, strict=True)` both works and gives fail-fast drift protection.
- Checkpoint (`~/models/sam-audio-large-tv/checkpoint.pt`, 14.9GB fp32) contains ONLY: `transformer.*` (11.72GB), `vision_encoder.*` (2.68GB), `audio_codec.*` (0.43GB), `align_masked_video.*`, `proj.*`, `memory_proj.*`, `embed_anchors.*`.
- fp16 safety (verified): RoPE applies rotations in fp32 (`rope.py:48` does `.float()` then `.type_as`); RMSNorm computes in fp32 (`transformer.py:46`); the timestep embedder computes `t.float() * freqs` and casts the result back to `t`'s dtype (`transformer.py:246-253`) — so the `time` tensor passed to `forward()` must be fp16; DACVAE has no registered buffers. Per-module `.half()` (params + buffers) is safe for all seven checkpoint-backed modules.
- `load_state_dict` with default `assign=False` uses `param.copy_(src)`, which converts fp32 mmap'd checkpoint tensors into fp16 params one tensor at a time — no RAM spike.
- `Batch` (`sam_audio/processor.py:58`) is a mutable plain class; reassigning `batch.audios` is fine.
- `worker/handlers/sam_audio_cleanup.py` imports `patch_sam_audio_model` and `process_audio_file` from `run_sam_interactive`, and `streamlit_app.py` calls the worker handler — so dtype fixes in `memory_safe_separate` automatically cover all three frontends. Only two `from_pretrained` call sites exist.
- Tests are unittest-style classes run via pytest: `.venv/bin/python -m pytest tests/ -v`.

## File Structure

- Create: `sam_audio_local/__init__.py` — empty package marker.
- Create: `sam_audio_local/loader.py` — config strip, checkpoint filter, vision stub, `load_sam_audio_optimized()`. One responsibility: building the memory-optimized model.
- Create: `tests/test_optimized_loader.py` — fast unit tests (no weights) + one skippable integration test (real weights).
- Modify: `run_sam_interactive.py` — dtype casts + rerank clamp in `memory_safe_separate`; loader swap at line 1163; rerank prompt note at line 1296.
- Modify: `worker/handlers/sam_audio_cleanup.py` — loader swap at line 64.
- Modify: `README.md` — short "memory-optimized local loading" section.

---

### Task 1: `sam_audio_local` package — pure helpers + vision stub (TDD, fast tests)

**Files:**
- Create: `sam_audio_local/__init__.py`
- Create: `sam_audio_local/loader.py`
- Test: `tests/test_optimized_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_optimized_loader.py`:

```python
from __future__ import annotations

import unittest

import torch

from sam_audio_local.loader import (
    StubVisionEncoder,
    filter_vision_keys,
    strip_ranker_config,
    stubbed_vision_encoder,
)


class _FakeVisionCfg:
    dim = 1024


class StripRankerConfigTests(unittest.TestCase):
    def test_rankers_set_to_none_and_rest_preserved(self):
        config = {
            "in_channels": 768,
            "visual_ranker": {"kind": "imagebind"},
            "text_ranker": {"kind": "ensemble"},
            "span_predictor": "pe-a-frame-large",
        }
        out = strip_ranker_config(config)
        self.assertIsNone(out["visual_ranker"])
        self.assertIsNone(out["text_ranker"])
        self.assertEqual(out["in_channels"], 768)
        self.assertEqual(out["span_predictor"], "pe-a-frame-large")

    def test_input_dict_not_mutated(self):
        config = {"visual_ranker": {"kind": "imagebind"}, "text_ranker": None}
        strip_ranker_config(config)
        self.assertEqual(config["visual_ranker"], {"kind": "imagebind"})


class FilterVisionKeysTests(unittest.TestCase):
    def test_drops_only_vision_encoder_keys(self):
        sd = {
            "vision_encoder.blocks.0.weight": torch.zeros(1),
            "transformer.layers.0.weight": torch.zeros(1),
            "audio_codec.encoder.weight": torch.zeros(1),
        }
        out = filter_vision_keys(sd)
        self.assertNotIn("vision_encoder.blocks.0.weight", out)
        self.assertIn("transformer.layers.0.weight", out)
        self.assertIn("audio_codec.encoder.weight", out)


class StubVisionEncoderTests(unittest.TestCase):
    def test_has_dim_and_no_parameters(self):
        stub = StubVisionEncoder(_FakeVisionCfg())
        self.assertEqual(stub.dim, 1024)
        self.assertEqual(sum(p.numel() for p in stub.parameters()), 0)

    def test_forward_raises(self):
        stub = StubVisionEncoder(_FakeVisionCfg())
        with self.assertRaises(RuntimeError):
            stub([torch.zeros(1, 3, 8, 8)])


class StubbedVisionEncoderContextTests(unittest.TestCase):
    def test_patches_and_restores(self):
        from sam_audio.model import model as sam_model_module

        original = sam_model_module.PerceptionEncoder
        with stubbed_vision_encoder():
            self.assertIs(sam_model_module.PerceptionEncoder, StubVisionEncoder)
        self.assertIs(sam_model_module.PerceptionEncoder, original)

    def test_restores_on_exception(self):
        from sam_audio.model import model as sam_model_module

        original = sam_model_module.PerceptionEncoder
        with self.assertRaises(ValueError):
            with stubbed_vision_encoder():
                raise ValueError("boom")
        self.assertIs(sam_model_module.PerceptionEncoder, original)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_optimized_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sam_audio_local'`

- [ ] **Step 3: Write the implementation**

Create `sam_audio_local/__init__.py` (empty file).

Create `sam_audio_local/loader.py`:

```python
"""Memory-optimized SAM-Audio loader for text-prompted separation.

Strips the vision encoder (PE-Core-L14, 2.7GB) and all rankers
(ImageBind/CLAP/Judge), mmap-loads the checkpoint to stay under the WSL2
RAM ceiling, and casts checkpoint-backed modules to fp16 (native on
Turing). Resident VRAM drops from ~31GB to ~7-8GB.

Design: docs/superpowers/specs/2026-06-11-rtx8000-memory-optimized-loader-design.md
"""

from __future__ import annotations

import contextlib
import json
import os

import torch

from sam_audio import SAMAudio
from sam_audio.model import model as _sam_model_module

# Checkpoint-backed modules cast to fp16. text_encoder (T5, fp16-fragile)
# and span_predictor (HF-loaded, ~1-2GB) stay fp32; dtype boundaries are
# handled in memory_safe_separate (run_sam_interactive.py).
FP16_MODULES = (
    "audio_codec",
    "transformer",
    "proj",
    "align_masked_video",
    "embed_anchors",
    "memory_proj",
    "timestep_emb",
)


def strip_ranker_config(config: dict) -> dict:
    """Return a copy of config.json contents with all rankers disabled."""
    out = dict(config)
    out["visual_ranker"] = None
    out["text_ranker"] = None
    return out


def filter_vision_keys(state_dict):
    """Drop vision_encoder weights from a checkpoint state dict."""
    return {
        k: v for k, v in state_dict.items() if not k.startswith("vision_encoder.")
    }


class StubVisionEncoder(torch.nn.Module):
    """Zero-parameter stand-in. Only `.dim` is read on the text-only path."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim

    def forward(self, videos):
        raise RuntimeError(
            "Vision encoder was stripped by the memory-optimized loader; "
            "video prompting is unavailable (text prompts only)."
        )


@contextlib.contextmanager
def stubbed_vision_encoder():
    """Make SAMAudio.__init__ build StubVisionEncoder instead of PE."""
    original = _sam_model_module.PerceptionEncoder
    _sam_model_module.PerceptionEncoder = StubVisionEncoder
    try:
        yield
    finally:
        _sam_model_module.PerceptionEncoder = original


def load_sam_audio_optimized(model_dir, device: str = "cuda") -> SAMAudio:
    """Load SAM-Audio for text-prompted separation in ~7-8GB VRAM.

    fp16 on CUDA; full fp32 on CPU fallback (fp16 CPU inference is
    unsupported for many ops).
    """
    model_dir = str(model_dir)
    with open(os.path.join(model_dir, "config.json")) as fin:
        config = strip_ranker_config(json.load(fin))

    cfg = SAMAudio.config_cls(**config)
    with stubbed_vision_encoder():
        model = SAMAudio(cfg)
    model.eval()

    if "cuda" in str(device):
        # Module.half() swaps tensors one at a time -- no RAM spike.
        for name in FP16_MODULES:
            getattr(model, name).half()

    state_dict = torch.load(
        os.path.join(model_dir, "checkpoint.pt"),
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    # SAMAudio.load_state_dict(strict=True) tolerates missing
    # text_encoder/span_predictor keys (HF-loaded) and raises on any other
    # mismatch -- fail-fast protection against checkpoint/package drift.
    model.load_state_dict(filter_vision_keys(state_dict), strict=True)
    del state_dict
    return model.to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_optimized_loader.py -v`
Expected: 7 passed (the `load_sam_audio_optimized` function is exercised in Task 2)

- [ ] **Step 5: Run the existing test suite to confirm nothing broke**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/ -v`
Expected: all pass (existing `test_queue_client.py`, `test_worker_core.py` unaffected)

- [ ] **Step 6: Commit**

```bash
cd /home/longboardfella/sam-audio
git add sam_audio_local/ tests/test_optimized_loader.py
git commit -m "feat: add memory-optimized loader package with vision stub and config strip"
```

---

### Task 2: Integration test for `load_sam_audio_optimized` (real weights)

**Files:**
- Modify: `tests/test_optimized_loader.py` (append)

- [ ] **Step 1: Append the integration test**

Append to `tests/test_optimized_loader.py`:

```python
from pathlib import Path

MODEL_DIR = Path.home() / "models" / "sam-audio-large-tv"


@unittest.skipUnless(
    (MODEL_DIR / "checkpoint.pt").exists(), "local model weights not present"
)
class LoadOptimizedIntegrationTests(unittest.TestCase):
    """Loads the real 14.9GB checkpoint -- takes a few minutes."""

    @classmethod
    def setUpClass(cls):
        from sam_audio_local.loader import load_sam_audio_optimized

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = load_sam_audio_optimized(MODEL_DIR, device=cls.device)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_vision_encoder_is_stub_with_no_parameters(self):
        from sam_audio_local.loader import StubVisionEncoder

        self.assertIsInstance(self.model.vision_encoder, StubVisionEncoder)
        self.assertEqual(
            sum(p.numel() for p in self.model.vision_encoder.parameters()), 0
        )

    def test_rankers_are_none(self):
        self.assertIsNone(self.model.visual_ranker)
        self.assertIsNone(self.model.text_ranker)

    def test_span_predictor_present(self):
        self.assertTrue(hasattr(self.model, "span_predictor"))

    def test_dtypes(self):
        if self.device != "cuda":
            self.skipTest("fp16 cast only applies on CUDA")
        self.assertEqual(
            next(self.model.transformer.parameters()).dtype, torch.float16
        )
        self.assertEqual(
            next(self.model.audio_codec.parameters()).dtype, torch.float16
        )
        self.assertEqual(
            next(self.model.text_encoder.parameters()).dtype, torch.float32
        )

    def test_resident_vram_under_10gb(self):
        if self.device != "cuda":
            self.skipTest("VRAM check requires CUDA")
        resident_gb = torch.cuda.memory_allocated() / 1e9
        self.assertLess(resident_gb, 10.0)
```

- [ ] **Step 2: Run the integration test**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -m pytest tests/test_optimized_loader.py -v --timeout=900 2>/dev/null || .venv/bin/python -m pytest tests/test_optimized_loader.py -v`
(use the plain second form if pytest-timeout is not installed)
Expected: all pass, including 5 integration tests. Note the printed VRAM number for the records.

If `torch.load(..., mmap=True)` raises (checkpoint not in zipfile format — unlikely for a Dec 2025 release): STOP and report; do not silently drop `mmap=True`, since that reintroduces the RAM spike.

- [ ] **Step 3: Commit**

```bash
cd /home/longboardfella/sam-audio
git add tests/test_optimized_loader.py
git commit -m "test: add real-weights integration test for optimized loader"
```

---

### Task 3: dtype boundaries + rerank clamp in `memory_safe_separate`

**Files:**
- Modify: `run_sam_interactive.py` (the `memory_safe_separate` function, lines ~296-431)

The model is now mixed-precision (codec/DiT fp16, T5/span-predictor fp32). Five boundary casts plus a rerank clamp. All casts are conditional/no-op when the model is fully fp32 (CPU fallback), so behaviour there is unchanged.

- [ ] **Step 1: Add rerank clamp and input cast at the top of `memory_safe_separate`**

Find (run_sam_interactive.py:314-316):

```python
    # Stage 1: Encode audio and prepare forward args
    forward_args = self._get_forward_args(batch, candidates=reranking_candidates)
```

Replace with:

```python
    if (
        reranking_candidates > 1
        and self.visual_ranker is None
        and self.text_ranker is None
    ):
        print(
            "Note: reranking unavailable (rankers stripped in memory-optimized "
            "build); using 1 candidate"
        )
        reranking_candidates = 1

    # fp16 boundary: optimized loader keeps codec/DiT in fp16, T5/spans in fp32
    model_dtype = next(self.audio_codec.parameters()).dtype
    if batch.audios.dtype != model_dtype:
        batch.audios = batch.audios.to(model_dtype)

    # Stage 1: Encode audio and prepare forward args
    forward_args = self._get_forward_args(batch, candidates=reranking_candidates)
    if forward_args["text_features"].dtype != model_dtype:
        forward_args["text_features"] = forward_args["text_features"].to(model_dtype)
```

- [ ] **Step 2: Cast span-predictor input to fp32**

Find (inside the `if predict_spans ...` block, run_sam_interactive.py:318-327):

```python
            audio_features=self._unrepeat_from_reranking(
                forward_args["audio_features"], reranking_candidates
            ),
```

Replace with:

```python
            audio_features=self._unrepeat_from_reranking(
                forward_args["audio_features"], reranking_candidates
            ).float(),
```

- [ ] **Step 3: Cast the ODE time tensor to the model dtype**

Find (inside `vector_field`, run_sam_interactive.py:~337-342):

```python
        res = self.forward(
            noisy_audio=noisy_audio,
            time=t.expand(noisy_audio.size(0)),
            **forward_args,
        )
```

Replace with:

```python
        res = self.forward(
            noisy_audio=noisy_audio,
            time=t.to(noisy_audio.dtype).expand(noisy_audio.size(0)),
            **forward_args,
        )
```

- [ ] **Step 4: Cast ODE output before decoding**

(`odeint` can promote the state to fp32 because the fp32 t-span participates in step arithmetic.)

Find (run_sam_interactive.py:~352):

```python
    generated_features = states[-1].transpose(1, 2)
```

Replace with:

```python
    generated_features = states[-1].transpose(1, 2).to(model_dtype)
```

- [ ] **Step 5: Return fp32 audio**

Find (the `SeparationResult` construction, run_sam_interactive.py:~424-429):

```python
    result = SeparationResult(
        target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
        residual=[
            wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
        ],
        noise=None,  # Already deleted to save memory
    )
```

Replace with:

```python
    result = SeparationResult(
        target=[wav[idx].float() for wav, idx in zip(target_wavs, idxs, strict=False)],
        residual=[
            wavs[idx].float() for wavs, idx in zip(residual_wavs, idxs, strict=False)
        ],
        noise=None,  # Already deleted to save memory
    )
```

- [ ] **Step 6: Sanity check — script still imports and fast tests pass**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -c "import run_sam_interactive" && .venv/bin/python -m pytest tests/ -v -k "not Integration"`
Expected: import succeeds; all fast tests pass. (Functional verification of the casts happens in Task 5's end-to-end run.)

- [ ] **Step 7: Commit**

```bash
cd /home/longboardfella/sam-audio
git add run_sam_interactive.py
git commit -m "feat: handle fp16/fp32 dtype boundaries and clamp rerank in memory_safe_separate"
```

---

### Task 4: Switch both load sites to the optimized loader

**Files:**
- Modify: `run_sam_interactive.py:34-36` (imports), `:1163` (load site), `:1296-1298` (rerank prompt)
- Modify: `worker/handlers/sam_audio_cleanup.py:18` (imports), `:64` (load site)

- [ ] **Step 1: Swap the interactive script's load site**

In `run_sam_interactive.py`, after the existing import block (line 34-36):

```python
from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.model.model import SeparationResult, DFLT_ODE_OPT
from sam_audio.processor import Batch
```

add:

```python
from sam_audio_local.loader import load_sam_audio_optimized
```

Then find (line ~1163):

```python
    model = SAMAudio.from_pretrained(str(model_dir)).eval().to(device)
```

Replace with:

```python
    model = load_sam_audio_optimized(model_dir, device)
```

(The loader already calls `.eval()` and `.to(device)`.)

- [ ] **Step 2: Mark reranking unavailable in the interactive prompt**

Find (line ~1296-1298):

```python
    rerank = get_user_input(
        "Reranking candidates (1-8)",
        config["rerank"],
```

Replace with:

```python
    rerank = get_user_input(
        "Reranking candidates (unavailable in memory-optimized build; use 1)",
        config["rerank"],
```

- [ ] **Step 3: Swap the worker handler's load site**

In `worker/handlers/sam_audio_cleanup.py`, find (line 18):

```python
from sam_audio import SAMAudio, SAMAudioProcessor
```

Replace with:

```python
from sam_audio import SAMAudioProcessor

from sam_audio_local.loader import load_sam_audio_optimized
```

Then find (line ~64):

```python
                model = SAMAudio.from_pretrained(model_dir_str).eval().to(device)
```

Replace with:

```python
                model = load_sam_audio_optimized(model_dir_str, device)
```

- [ ] **Step 4: Verify imports and fast tests**

Run: `cd /home/longboardfella/sam-audio && .venv/bin/python -c "import run_sam_interactive; import worker.handlers.sam_audio_cleanup" && .venv/bin/python -m pytest tests/ -v -k "not Integration"`
Expected: imports succeed, fast tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/longboardfella/sam-audio
git add run_sam_interactive.py worker/handlers/sam_audio_cleanup.py
git commit -m "feat: use memory-optimized loader in interactive script and queue worker"
```

---

### Task 5: End-to-end verification on the RTX 8000

**Files:** none created (verification only; logs land in `/tmp/`)

- [ ] **Step 1: Pick a real input file**

Run: `ls -lh /mnt/f/hf-home/audio_input/ | head`
Pick the smallest real audio file; call it `$INPUT` below. If the directory is empty, ask the user for a test file.

- [ ] **Step 2: Start VRAM + RAM logging**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > /tmp/sam_fp16_vram.log &
echo $! > /tmp/sam_vram_pid
free -m -s 5 | grep ^Mem > /tmp/sam_fp16_ram.log &
echo $! > /tmp/sam_ram_pid
```

- [ ] **Step 3: Run the smoke test with the user's production settings**

```bash
cd /home/longboardfella/sam-audio
.venv/bin/python colab_smoke_test.py \
  --input "$INPUT" \
  --model-dir ~/models/sam-audio-large-tv \
  --description "men's voices" \
  --chunk-duration 30 --overlap 2 \
  --rerank 1 --predict-spans \
  --memory-fraction 0.85 --device cuda \
  --output-dir ./audio_output/fp16_smoke
```

Expected: completes without CUDA OOM and without the WSL2 OOM-killer; `audio_output/fp16_smoke/` contains a result zip with `target.wav` and `residual.wav`.

- [ ] **Step 4: Stop logging and check the numbers**

```bash
kill $(cat /tmp/sam_vram_pid) $(cat /tmp/sam_ram_pid)
sort -n /tmp/sam_fp16_vram.log | tail -1   # peak VRAM (MiB)
awk '{print $3}' /tmp/sam_fp16_ram.log | sort -n | tail -1  # peak RAM used (MB)
dmesg 2>/dev/null | tail -5 | grep -i "killed process" || echo "no OOM-killer events"
```

Expected: peak VRAM well under 46080 MiB (anticipate ~10-20GB including activations); peak RAM under 23GB; no OOM-killer events.

- [ ] **Step 5: Output sanity check**

```bash
.venv/bin/python - <<'EOF'
import glob, zipfile
zips = glob.glob("audio_output/fp16_smoke/**/*.zip", recursive=True)
assert zips, "no result zip produced"
with zipfile.ZipFile(zips[0]) as z:
    names = z.namelist()
print(names)
assert any("target" in n for n in names) and any("residual" in n for n in names)
EOF
```

Then report results to the user with the peak VRAM/RAM numbers and ask them to listen to `target.wav`/`residual.wav` for fp16 artifacts (compare against a known-good Colab output if available). **Quality sign-off is the user's call — do not claim quality success yourself.**

- [ ] **Step 6: If quality is degraded (user reports artifacts)**

Fallback knob, in order: remove `"audio_codec"` from `FP16_MODULES` in `sam_audio_local/loader.py` (decoder in fp32, +0.2GB) and add a `.to(model_dtype)`-style cast guard where `generated_features` feeds `audio_codec.decode` in `memory_safe_separate` — i.e. change `.to(model_dtype)` from Task 3 Step 4 to `.to(next(self.audio_codec.parameters()).dtype)`. Re-run Step 3. Only do this if the user reports artifacts.

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a section to README.md after the "## Features" block**

```markdown
## Memory-Optimized Local Loading (RTX 8000 / 48GB)

`sam_audio_local/loader.py` loads SAM-Audio large in ~7-8GB resident VRAM
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
```

- [ ] **Step 2: Commit**

```bash
cd /home/longboardfella/sam-audio
git add README.md
git commit -m "docs: describe memory-optimized local loading"
```

---

## Self-review notes

- Spec coverage: ranker strip (Task 1), vision stub (Task 1), mmap + fail-fast key check (Task 1/2), fp16 cast with fp32 T5 (Task 1), dtype boundaries in `memory_safe_separate` (Task 3), call-site swaps + rerank pinning (Task 4), CPU fp32 fallback (loader's `"cuda" in device` guard), unit + e2e + quality verification (Tasks 1/2/5), fp16-drift fallback knob (Task 5 Step 6). All spec sections have tasks.
- README.md has uncommitted user changes in the working tree — commit only the files named in each task (`git add` specific paths, never `git add -A`).
