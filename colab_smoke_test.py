#!/usr/bin/env python3
"""Run a single-file SAM-Audio cleanup smoke test (Colab-friendly, no Streamlit)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from worker.handlers.sam_audio_cleanup import handle


def _progress(pct: float, msg: str, stage: Optional[str]) -> None:
    s = stage or "-"
    print(f"[{pct:6.2f}%] {s:>10s} | {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one SAM-Audio cleanup test job")
    ap.add_argument("--input", required=True, help="Input audio file path")
    ap.add_argument("--model-dir", required=True, help="Local SAM-Audio model directory")
    ap.add_argument("--description", default="speech", help="Target description")
    ap.add_argument("--output-dir", default="./colab_output", help="Where to write result zip/wavs")

    # Conservative defaults to reduce OOM risk on shared GPUs.
    ap.add_argument("--chunk-duration", type=float, default=12.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--rerank", type=int, default=1)
    ap.add_argument("--predict-spans", action="store_true")
    ap.add_argument("--memory-fraction", type=float, default=0.60)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--allow-cpu-fallback", action="store_true", default=True)
    ap.add_argument("--no-cpu-fallback", action="store_false", dest="allow_cpu_fallback")
    ap.add_argument("--trial-seconds", type=int, default=0)
    ap.add_argument("--normalize-percent", type=float, default=0.0)
    ap.add_argument("--output-sample-rate", type=int, default=0)
    ap.add_argument("--output-channels", type=int, default=0)

    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    payload = {
        "description": args.description,
        "model_dir": str(model_dir),
        "chunk_duration": float(args.chunk_duration),
        "overlap": float(args.overlap),
        "rerank": int(args.rerank),
        "predict_spans": bool(args.predict_spans),
        "memory_fraction": float(args.memory_fraction),
        "device": args.device,
        "allow_cpu_fallback": bool(args.allow_cpu_fallback),
        "trial_seconds": int(args.trial_seconds),
        "normalize_percent": float(args.normalize_percent),
    }

    if args.output_sample_rate > 0:
        payload["output_sample_rate"] = int(args.output_sample_rate)
    if args.output_channels > 0:
        payload["output_channels"] = int(args.output_channels)

    # Keep a stable local copy of output artifacts.
    with tempfile.TemporaryDirectory(prefix="sam_colab_smoke_") as td:
        job = {
            "id": 1,
            "type": "sam_audio_cleanup",
            "input_filename": input_path.name,
        }

        result = handle(
            input_path=input_path,
            input_data=payload,
            job=job,
            progress_cb=_progress,
        )

        zip_src = Path(result["output_file"]).resolve()
        zip_dst = out_dir / "audio_cleanup_result.zip"
        shutil.copy2(zip_src, zip_dst)

    print("\n=== COMPLETE ===")
    print(f"Result zip: {zip_dst}")
    print("Metadata:")
    print(json.dumps(result["output_data"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
