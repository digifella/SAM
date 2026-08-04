"""Spectral-subtraction denoiser.

SAM-Audio is source separation and fails on single-speaker broadband noise.
This is the denoiser for that case. Pure numpy/scipy: no torch, no GPU, no model.
"""
from __future__ import annotations

import numpy as np

SILENCE_GATE_DBFS = -90.0
NOISE_WINDOW_SECONDS = 3.0
MIN_NOISE_HEADROOM_DB = 6.0
MAX_NOISE_DRIFT_DB = 10.0
STFT_WINDOW = 2048
STFT_OVERLAP = 1536
STRENGTH_PRESETS = {
    "gentle": (1.5, 0.10),
    "normal": (2.0, 0.05),
    "aggressive": (3.0, 0.02),
}

_FLOOR = 1e-12  # clamp so digital silence gives a finite dBFS, not -inf


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim > 1 else x


def frame_rms_dbfs(x: np.ndarray, sr: int, frame_seconds: float = 0.25) -> np.ndarray:
    """Per-frame RMS in dBFS. Silent frames clamp to a finite floor, never -inf."""
    x = _mono(x)
    n = max(1, int(frame_seconds * sr))
    count = len(x) // n
    if count == 0:
        return np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), _FLOOR))])
    frames = x[:count * n].reshape(count, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, _FLOOR))


def silence_mask(x: np.ndarray, sr: int, frame_seconds: float = 0.25) -> np.ndarray:
    """True where a frame is digital silence rather than quiet noise.

    Excluding these is what stops the noise profile being estimated from zeros.
    """
    return frame_rms_dbfs(x, sr, frame_seconds) <= SILENCE_GATE_DBFS
