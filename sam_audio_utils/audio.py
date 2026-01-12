from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch


def first_tensor(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def tensor_to_numpy(value):
    tensor = first_tensor(value).detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(torch.float32).transpose(0, 1).contiguous().numpy()


def run_ffmpeg_convert(
    in_path: Path,
    out_path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_bin: str = "ffmpeg",
):
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(
            f"ffmpeg binary '{ffmpeg_bin}' not found on PATH; install ffmpeg or set SAM_FFMPEG_BIN"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(in_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(out_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        stderr_tail = proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:]
        raise RuntimeError(f"ffmpeg convert failed for {in_path}:\n{stderr_tail}")
    return out_path
