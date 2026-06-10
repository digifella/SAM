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
