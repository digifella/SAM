"""Memory-safe SAM-Audio with patched reranking for WSL2.

This script monkey-patches SAMAudio.separate() to add aggressive memory cleanup
between ODE generation and reranking steps, preventing WSL2 crashes.

Key improvements:
1. Move generated waveforms to CPU before reranking
2. CUDA synchronization and cache clearing between heavy operations
3. CLAP ranker processes on CPU to avoid GPU memory spike
"""

import os
from pathlib import Path
import argparse
import gc
import sys
import torch
import soundfile as sf
from typing import Any, Dict, Optional

# Set PyTorch memory allocator settings BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.model.model import SeparationResult, DFLT_ODE_OPT
from sam_audio.processor import Batch
from sam_audio_utils import tensor_to_numpy, env_path, run_ffmpeg_convert
from torchdiffeq import odeint

DEFAULT_INPUT_DIR = env_path("SAM_INPUT_DIR", Path("./audio_input"))
DEFAULT_OUTPUT_DIR = env_path("SAM_OUTPUT_DIR", Path("./audio_output"))
DEFAULT_CACHE_DIR = env_path("SAM_CACHE_DIR", Path("./audio_cache"))
DEFAULT_MODEL_DIR = env_path(
    "SAM_MODEL_DIR", Path.home() / "models" / "sam-audio-large-tv"
)
DEFAULT_FFMPEG_BIN = os.environ.get("SAM_FFMPEG_BIN", "ffmpeg")


def aggressive_cleanup():
    """Perform comprehensive memory cleanup with proper ordering."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # CRITICAL: wait for GPU ops to finish
        torch.cuda.empty_cache()
    gc.collect()


def log_memory(stage: str, verbose: bool = True):
    """Log current memory state."""
    if not verbose or not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"[{stage}] GPU Memory: {allocated:.2f}GB allocated, "
          f"{reserved:.2f}GB reserved, {total - allocated:.2f}GB free of {total:.2f}GB")


def memory_safe_separate(
    self,
    batch: Batch,
    noise: Optional[torch.Tensor] = None,
    ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
    reranking_candidates: int = 1,
    predict_spans: bool = False,
    verbose: bool = False,
) -> SeparationResult:
    """Memory-safe version of SAMAudio.separate() with cleanup between stages.

    This patched version adds aggressive memory cleanup between:
    1. ODE generation
    2. Waveform decoding
    3. Reranking

    It also moves tensors to CPU before reranking to prevent GPU OOM.
    """

    # Stage 1: Encode audio and prepare forward args
    log_memory("Before encoding", verbose)
    forward_args = self._get_forward_args(batch, candidates=reranking_candidates)

    if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
        batch = self.predict_spans(
            batch=batch,
            audio_features=self._unrepeat_from_reranking(
                forward_args["audio_features"], reranking_candidates
            ),
            audio_pad_mask=self._unrepeat_from_reranking(
                forward_args["audio_pad_mask"], reranking_candidates
            ),
        )

    audio_features = forward_args["audio_features"]
    B, T, C = audio_features.shape
    C = C // 2  # we stack audio_features, so the actual channels is half

    if noise is None:
        noise = torch.randn_like(audio_features)

    # Stage 2: ODE integration (heavy GPU operation)
    log_memory("Before ODE", verbose)

    def vector_field(t, noisy_audio):
        res = self.forward(
            noisy_audio=noisy_audio,
            time=t.expand(noisy_audio.size(0)),
            **forward_args,
        )
        return res

    states = odeint(
        vector_field,
        noise,
        torch.tensor([0.0, 1.0], device=noise.device),
        **ode_opt,
    )

    generated_features = states[-1].transpose(1, 2)

    # CRITICAL: Clean up ODE solver memory before decoding
    del states, noise
    aggressive_cleanup()
    log_memory("After ODE (before decode)", verbose)

    # Stage 3: Decode waveforms
    # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
    wavs = self.audio_codec.decode(generated_features.reshape(2 * B, C, T)).view(
        B, 2, -1
    )

    # CRITICAL: Move wavs to CPU immediately to free GPU memory before reranking
    device_for_reranking = wavs.device
    wavs_cpu = wavs.cpu()
    del wavs, generated_features
    aggressive_cleanup()
    log_memory("After decode (wavs on CPU)", verbose)

    # Stage 4: Unbatch on CPU
    bsz = wavs_cpu.size(0) // reranking_candidates
    sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)

    target_wavs = self.unbatch(
        wavs_cpu[:, 0].view(bsz, reranking_candidates, -1), sizes.cpu()
    )
    residual_wavs = self.unbatch(
        wavs_cpu[:, 1].view(bsz, reranking_candidates, -1), sizes.cpu()
    )

    del wavs_cpu
    gc.collect()

    # Stage 5: Reranking (if needed)
    if (
        reranking_candidates > 1
        and batch.masked_video is not None
        and self.visual_ranker is not None
    ):
        log_memory("Before visual reranking", verbose)
        scores = self.visual_ranker(
            extracted_audio=target_wavs,
            videos=batch.masked_video,
            sample_rate=self.audio_codec.sample_rate,
        )
        idxs = scores.argmax(dim=1)
        del scores
        aggressive_cleanup()
        log_memory("After visual reranking", verbose)

    elif reranking_candidates > 1 and self.text_ranker is not None:
        log_memory("Before text reranking", verbose)

        # Get the device the ranker expects
        ranker_device = next(self.text_ranker.parameters()).device

        # Prepare input audio on the ranker's device
        input_audio = [
            audio[:, :size].expand(reranking_candidates, -1).to(ranker_device)
            for audio, size in zip(batch.audios, sizes, strict=False)
        ]

        # Move target_wavs to ranker device temporarily for scoring
        target_wavs_device = [wav.to(ranker_device) for wav in target_wavs]

        # Run CLAP ranker
        scores = self.text_ranker(
            extracted_audio=target_wavs_device,
            input_audio=input_audio,
            descriptions=batch.descriptions,
            sample_rate=self.audio_codec.sample_rate,
        )
        idxs = scores.argmax(dim=1).cpu()  # Move indices to CPU

        # Clean up device tensors
        del scores, input_audio, target_wavs_device
        aggressive_cleanup()

        # Keep target_wavs on CPU (already there from earlier)
        log_memory("After text reranking", verbose)
    else:
        idxs = torch.zeros(bsz, dtype=torch.long)

    # Stage 6: Select best candidates and return to device if needed
    result = SeparationResult(
        target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
        residual=[
            wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
        ],
        noise=None,  # Already deleted to save memory
    )

    log_memory("After separate complete", verbose)
    return result


def patch_sam_audio_model(model: SAMAudio, verbose: bool = False):
    """Monkey-patch the SAMAudio model with memory-safe separate method.

    Args:
        model: SAMAudio model instance to patch
        verbose: Whether to enable verbose memory logging
    """
    import types

    # Create a closure to capture verbose flag
    def make_separate(verbose_flag):
        def separate_wrapper(self, batch, noise=None, ode_opt=DFLT_ODE_OPT,
                           reranking_candidates=1, predict_spans=False):
            return memory_safe_separate(
                self, batch, noise, ode_opt, reranking_candidates,
                predict_spans, verbose=verbose_flag
            )
        return separate_wrapper

    # Replace the separate method
    model.separate = types.MethodType(make_separate(verbose), model)
    print("✓ Model patched with memory-safe reranking")


def main():
    ap = argparse.ArgumentParser(description="Memory-safe SAM-Audio with patched reranking")
    ap.add_argument("--file", required=True, help="Input WAV filename or path")
    ap.add_argument(
        "--desc", required=True, help="Text description (e.g. 'young boy speaking')"
    )
    ap.add_argument("--rerank", type=int, default=2, help="Reranking candidates (>=1) [default: 2 - safe with patch]")
    ap.add_argument(
        "--spans", action="store_true", help="Enable speech span prediction"
    )
    ap.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory with SAM-Audio weights",
    )
    ap.add_argument(
        "--input_dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Base directory for audio inputs",
    )
    ap.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write separated tracks",
    )
    ap.add_argument(
        "--cache_dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for converted mono 16k WAVs",
    )
    ap.add_argument(
        "--ffmpeg_bin",
        default=DEFAULT_FFMPEG_BIN,
        help="ffmpeg executable to use for conversion",
    )
    ap.add_argument(
        "--convert",
        dest="convert",
        default=True,
        action="store_true",
        help="Convert to mono 16 kHz before inference (default: on)",
    )
    ap.add_argument(
        "--no-convert",
        dest="convert",
        action="store_false",
        help="Disable mono/16 kHz conversion",
    )
    ap.add_argument(
        "--convert_sr",
        type=int,
        default=16000,
        help="Sample rate for conversion",
    )
    ap.add_argument(
        "--force_reconvert",
        action="store_true",
        help="Ignore cached conversions",
    )
    ap.add_argument(
        "--memory_fraction",
        type=float,
        default=0.7,
        help="Max fraction of GPU memory to use (0.0-1.0) [default: 0.7]",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose memory logging",
    )
    args = ap.parse_args()

    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    input_path = Path(args.file)
    if not input_path.is_absolute():
        input_path = input_dir / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.convert:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAM-Audio Memory-Safe Patched Processing")
    print(f"{'='*70}")
    print(f"Input: {input_path.name}")
    print(f"Description: {args.desc}")
    print(f"Rerank: {args.rerank}, Spans: {args.spans}")

    # Configure device and memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
        print(f"GPU memory fraction limit: {args.memory_fraction}")
        log_memory("Initial", args.verbose)

    # Initial cleanup
    aggressive_cleanup()

    model = None
    processor = None
    batch = None

    try:
        # Load processor
        if args.verbose:
            print("\n[Stage 1] Loading processor...")
        processor = SAMAudioProcessor.from_pretrained(args.model_dir)
        aggressive_cleanup()
        log_memory("After processor load", args.verbose)

        # Load model
        if args.verbose:
            print("\n[Stage 2] Loading model...")
        model = SAMAudio.from_pretrained(args.model_dir).eval().to(device)

        # PATCH THE MODEL
        patch_sam_audio_model(model, verbose=args.verbose)

        aggressive_cleanup()
        log_memory("After model load & patch", args.verbose)

        # Audio conversion (if needed)
        proc_path = input_path
        if args.convert:
            proc_path = cache_dir / f"{input_path.stem}_mono{args.convert_sr}.wav"
            if args.force_reconvert or not proc_path.exists():
                if args.verbose:
                    print(f"\n[Stage 3] Converting audio...")
                print(f"Converting {input_path.name} -> {proc_path.name} ({args.convert_sr} Hz mono)")
                run_ffmpeg_convert(
                    input_path,
                    proc_path,
                    sample_rate=args.convert_sr,
                    channels=1,
                    ffmpeg_bin=args.ffmpeg_bin,
                )
            else:
                print(f"Using cached conversion -> {proc_path.name}")

        # Process audio
        if args.verbose:
            print("\n[Stage 4] Processing audio...")
        batch = processor(audios=[str(proc_path)], descriptions=[args.desc]).to(device)
        aggressive_cleanup()
        log_memory("After batch preparation", args.verbose)

        # Run inference with patched method
        if args.verbose:
            print("\n[Stage 5] Running inference with memory-safe reranking...")

        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=args.spans,
                reranking_candidates=max(1, args.rerank)
            )

        log_memory("After inference", args.verbose)

        # Save results
        if args.verbose:
            print("\n[Stage 6] Saving results...")

        sr = int(processor.audio_sampling_rate)
        out_base = input_path.stem
        target_path = output_dir / f"{out_base}_target.wav"
        residual_path = output_dir / f"{out_base}_residual.wav"

        # Convert to numpy (results already on CPU from patched method)
        target_numpy = tensor_to_numpy(result.target)
        residual_numpy = tensor_to_numpy(result.residual)

        # Clear result
        del result
        aggressive_cleanup()

        # Write files
        sf.write(target_path, target_numpy, sr, subtype="PCM_16")
        sf.write(residual_path, residual_numpy, sr, subtype="PCM_16")

        print(f"\n{'='*70}")
        print(f"SUCCESS! Output files:")
        print(f"  Target:   {target_path}")
        print(f"  Residual: {residual_path}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Reranking: {args.rerank} candidates (memory-safe)")
        log_memory("Final", args.verbose)
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        log_memory("Error state", args.verbose)
        raise

    finally:
        # Comprehensive cleanup
        if args.verbose:
            print("\n[Cleanup] Releasing all resources...")

        if batch is not None:
            del batch
        if model is not None:
            del model
        if processor is not None:
            del processor

        aggressive_cleanup()
        log_memory("After final cleanup", args.verbose)


if __name__ == "__main__":
    main()
