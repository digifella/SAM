#!/usr/bin/env python3
"""
SAM-Audio Interactive Batch CLI with Chunking Support and Memory-Safe Reranking
- Processes all audio files in input directory
- Converts to mono 16kHz before processing (optional)
- Handles large audio files by processing in chunks
- Auto-increments output filenames if they exist
- Saves user preferences for future runs
- Memory-safe reranking patch for WSL2 stability
"""

# IMPORTANT: Set CUDA memory config BEFORE importing torch
import os
# Optimized for high-VRAM GPUs (24GB+) - allows larger contiguous allocations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import logging
import readline  # Enable arrow keys and line editing in input()
import subprocess
import sys
import tempfile
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import torch
from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.model.model import SeparationResult, DFLT_ODE_OPT
from sam_audio.processor import Batch
from torchdiffeq import odeint

# Default configuration file
CONFIG_FILE = Path.home() / ".sam_audio_config.json"

# Log file - timestamped to preserve crash logs
LOG_DIR = Path.home() / ".sam_audio_logs"
LOG_DIR.mkdir(exist_ok=True)

# Supported audio formats
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

# Safety limits
MAX_FILES_PER_SESSION = 100  # Prevent runaway processing

# Default values
DEFAULT_CONFIG = {
    "model_dir": str(Path.home() / "models" / "sam-audio-large-tv"),
    "input_dir": "./audio_input",
    "output_dir": "./audio_output",
    "convert_to_mono16k": True,
    "memory_fraction": 0.85,  # Higher for high-VRAM GPUs (24GB+)
    "rerank": 1,
    "predict_spans": False,
    "chunk_duration": 60,  # seconds per chunk
    "overlap": 2.0,  # seconds of overlap between chunks
    "last_description": "speech",
}


# ============================================================================
# Memory Monitoring and Logging
# ============================================================================

def setup_logger(log_file: Path) -> logging.Logger:
    """Setup logger that writes to both file and console."""
    logger = logging.getLogger('sam_audio_monitor')
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers.clear()

    # File handler - always write everything
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler - only important messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    return logger


def log_memory_stats(logger: logging.Logger, stage: str, device: str = "cuda"):
    """Log detailed memory statistics at a specific stage."""
    # CPU memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]  # Get numeric value
                    meminfo[key] = int(value)

        mem_total_gb = meminfo.get('MemTotal', 0) / (1024**2)
        mem_available_gb = meminfo.get('MemAvailable', 0) / (1024**2)
        mem_used_gb = mem_total_gb - mem_available_gb
        mem_percent = (mem_used_gb / mem_total_gb * 100) if mem_total_gb > 0 else 0

        cpu_msg = f"CPU: {mem_used_gb:.2f}/{mem_total_gb:.2f} GB ({mem_percent:.1f}%)"
    except Exception as e:
        cpu_msg = f"CPU: Error reading meminfo - {e}"

    # GPU memory
    if device == "cuda" and torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_percent = (allocated / total * 100) if total > 0 else 0

            gpu_msg = f"GPU: {allocated:.2f}/{total:.2f} GB ({gpu_percent:.1f}%), Reserved: {reserved:.2f} GB"
        except Exception as e:
            gpu_msg = f"GPU: Error - {e}"
    else:
        gpu_msg = "GPU: Not available"

    logger.info(f"{stage:40s} | {cpu_msg:50s} | {gpu_msg}")

    # Force flush to ensure log is written even if we crash
    for handler in logger.handlers:
        handler.flush()


# ============================================================================
# Memory-Safe Reranking Patch for WSL2
# ============================================================================

def aggressive_cleanup():
    """Perform comprehensive memory cleanup with proper ordering."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # CRITICAL: wait for GPU ops to finish
        torch.cuda.empty_cache()
    gc.collect()


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

    # CRITICAL: Synchronize before cleanup to ensure ODE operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # CRITICAL: Clean up ODE solver memory before decoding
    del states, noise
    aggressive_cleanup()

    # Stage 3: Decode waveforms
    wavs = self.audio_codec.decode(generated_features.reshape(2 * B, C, T)).view(
        B, 2, -1
    )

    # CRITICAL: Move wavs to CPU immediately to free GPU memory before reranking
    wavs_cpu = wavs.cpu().detach()  # Detach to break computational graph
    del wavs, generated_features, forward_args
    aggressive_cleanup()

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
        scores = self.visual_ranker(
            extracted_audio=target_wavs,
            videos=batch.masked_video,
            sample_rate=self.audio_codec.sample_rate,
        )
        idxs = scores.argmax(dim=1)
        del scores
        aggressive_cleanup()

    elif reranking_candidates > 1 and self.text_ranker is not None:
        # Prepare input audio on CPU
        input_audio = [
            audio[:, :size].expand(reranking_candidates, -1).cpu()
            for audio, size in zip(batch.audios, sizes, strict=False)
        ]

        # CRITICAL: Run CLAP ranker with CPU tensors
        scores = self.text_ranker(
            extracted_audio=target_wavs,
            input_audio=input_audio,
            descriptions=batch.descriptions,
            sample_rate=self.audio_codec.sample_rate,
        )
        idxs = scores.argmax(dim=1)

        del scores, input_audio
        aggressive_cleanup()
    else:
        idxs = torch.zeros(bsz, dtype=torch.long)

    # Stage 6: Select best candidates and return
    result = SeparationResult(
        target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
        residual=[
            wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
        ],
        noise=None,  # Already deleted to save memory
    )

    return result


def patch_sam_audio_model(model: SAMAudio):
    """Monkey-patch the SAMAudio model with memory-safe separate method."""
    def separate_wrapper(self, batch, noise=None, ode_opt=DFLT_ODE_OPT,
                       reranking_candidates=1, predict_spans=False):
        return memory_safe_separate(
            self, batch, noise, ode_opt, reranking_candidates, predict_spans
        )

    # Replace the separate method
    model.separate = types.MethodType(separate_wrapper, model)
    print("✓ Model patched with memory-safe reranking for WSL2")


# ============================================================================
# End of Memory-Safe Reranking Patch
# ============================================================================


def load_config() -> Dict[str, Any]:
    """Load saved configuration or return defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults to add any new keys
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"Warning: Could not load config from {CONFIG_FILE}: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]):
    """Save configuration for future runs."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"Warning: Could not save config: {e}")


def get_user_input(prompt: str, default: Any, input_type=str) -> Any:
    """Get user input with default value pre-filled for easy editing."""
    default_str = str(default)

    # Pre-fill the input with the default value for easy editing
    def pre_fill_default():
        readline.insert_text(default_str)
        readline.redisplay()

    readline.set_pre_input_hook(pre_fill_default)

    try:
        user_input = input(f"{prompt}: ").strip()
    finally:
        # Clear the hook after use
        readline.set_pre_input_hook()

    if not user_input:
        return default

    if input_type == bool:
        return user_input.lower() in ('y', 'yes', 't', 'true', '1')
    elif input_type == int:
        try:
            return int(user_input)
        except ValueError:
            print(f"Invalid integer, using default: {default}")
            return default
    elif input_type == float:
        try:
            return float(user_input)
        except ValueError:
            print(f"Invalid float, using default: {default}")
            return default
    else:
        return user_input


def find_audio_files(input_dir: Path) -> List[Path]:
    """Find all audio files in input directory, excluding SAM-Audio output files."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(input_dir.glob(f'*{ext}'))

    # CRITICAL FIX: Exclude files that are outputs from THIS script
    # Only skip files that END with _target or _residual (SAM-Audio outputs)
    # Don't skip files that merely contain these words in the middle of their names
    filtered_files = []
    for f in audio_files:
        stem_lower = f.stem.lower()
        # Check if filename ends with _target or _residual (before extension)
        if stem_lower.endswith('_target') or stem_lower.endswith('_residual'):
            print(f"  ⊗ Skipping SAM-Audio output file: {f.name}")
            continue
        filtered_files.append(f)

    return sorted(filtered_files)


def get_unique_output_path(output_dir: Path, stem: str, suffix: str = '.wav') -> Path:
    """Get unique output path by incrementing number if file exists."""
    base_path = output_dir / f"{stem}{suffix}"
    if not base_path.exists():
        return base_path

    # File exists, increment
    counter = 1
    while True:
        new_path = output_dir / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def convert_to_mono_16k(input_path: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    """
    Convert audio to mono 16kHz WAV using ffmpeg.
    Returns path to converted file.
    """
    # Create temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="sam_mono16k_")
    os.close(fd)
    output_path = Path(tmp_path)

    cmd = [
        ffmpeg_bin,
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e.stderr.decode()}")
        raise
    except subprocess.TimeoutExpired:
        print("Error: ffmpeg conversion timed out")
        raise


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(audio_path)
    return info.duration


def chunk_audio_generator(audio_path: Path, chunk_duration: float, overlap: float):
    """
    Generator that yields audio chunks one at a time to minimize memory usage.
    Yields (start_time, end_time, chunk_data, sample_rate) tuples.
    """
    info = sf.info(audio_path)
    total_duration = info.duration
    sample_rate = info.samplerate

    start = 0.0
    step = chunk_duration - overlap  # Non-overlapping portion

    while start < total_duration:
        end = min(start + chunk_duration, total_duration)

        # Read chunk
        start_frame = int(start * sample_rate)
        num_frames = int((end - start) * sample_rate)

        audio_data, sr = sf.read(
            audio_path,
            start=start_frame,
            frames=num_frames,
            always_2d=False
        )

        yield (start, end, audio_data, sr)

        # Move to next chunk (advance by non-overlapping portion)
        start += step


def count_chunks(audio_path: Path, chunk_duration: float, overlap: float) -> int:
    """
    Calculate the number of chunks without loading audio into memory.
    """
    info = sf.info(audio_path)
    total_duration = info.duration

    count = 0
    start = 0.0
    step = chunk_duration - overlap  # Non-overlapping portion

    while start < total_duration:
        count += 1
        start += step

    return count


def process_chunk(
    chunk_data: np.ndarray,
    sample_rate: int,
    description: str,
    model: SAMAudio,
    processor: SAMAudioProcessor,
    device: str,
    rerank: int,
    predict_spans: bool,
    logger: Optional[logging.Logger] = None,
    chunk_idx: int = 0,
) -> tuple:
    """Process a single audio chunk with aggressive memory cleanup."""
    if logger:
        log_memory_stats(logger, f"Chunk {chunk_idx}: Start", device)

    # Save chunk to temp file (SAM-Audio expects file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(tmp_path, chunk_data, sample_rate)

    batch = None
    result = None

    try:
        # Process chunk
        batch = processor(audios=[str(tmp_path)], descriptions=[description]).to(device)
        if logger:
            log_memory_stats(logger, f"Chunk {chunk_idx}: After batch load", device)

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
            result = model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=max(1, int(rerank)),
            )

        if logger:
            log_memory_stats(logger, f"Chunk {chunk_idx}: After inference", device)

        # CRITICAL FIX: result.target/residual are lists, extract first element
        # Convert to numpy and explicitly copy to ensure contiguous memory
        target_tensor = result.target[0] if isinstance(result.target, list) else result.target
        residual_tensor = result.residual[0] if isinstance(result.residual, list) else result.residual

        target = target_tensor.squeeze().cpu().numpy().copy()
        residual = residual_tensor.squeeze().cpu().numpy().copy()

        # Clean up GPU/torch memory FIRST
        del batch, result, target_tensor, residual_tensor
        aggressive_cleanup()

        if logger:
            log_memory_stats(logger, f"Chunk {chunk_idx}: After cleanup", device)

        return target, residual

    except Exception as e:
        # Ensure cleanup happens even on error
        if logger:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
            log_memory_stats(logger, f"Chunk {chunk_idx}: Error state", device)

        # Clean up any allocated tensors
        if batch is not None:
            del batch
        if result is not None:
            del result
        aggressive_cleanup()
        raise

    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)
        # Force CPU memory cleanup
        gc.collect()


def merge_chunks(chunks: list, overlap: float, sample_rate: int) -> np.ndarray:
    """
    Merge overlapping chunks with crossfade.
    chunks: list of numpy arrays
    """
    if len(chunks) == 1:
        return chunks[0]

    overlap_samples = int(overlap * sample_rate)

    # Calculate total length
    total_samples = sum(len(c) for c in chunks) - (len(chunks) - 1) * overlap_samples
    merged = np.zeros(total_samples, dtype=chunks[0].dtype)

    pos = 0
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk - no crossfade at start
            merged[:len(chunk)] = chunk
            pos = len(chunk) - overlap_samples
        else:
            # Crossfade with previous chunk
            if overlap_samples > 0:
                fade_out = np.linspace(1, 0, overlap_samples)
                fade_in = np.linspace(0, 1, overlap_samples)

                # Apply crossfade
                merged[pos:pos + overlap_samples] = (
                    merged[pos:pos + overlap_samples] * fade_out +
                    chunk[:overlap_samples] * fade_in
                )

                # Add rest of chunk
                merged[pos + overlap_samples:pos + len(chunk)] = chunk[overlap_samples:]
            else:
                merged[pos:pos + len(chunk)] = chunk

            pos += len(chunk) - overlap_samples

    return merged


def process_audio_file(
    audio_path: Path,
    description: str,
    output_dir: Path,
    model: SAMAudio,
    processor: SAMAudioProcessor,
    device: str,
    memory_fraction: float,
    rerank: int,
    predict_spans: bool,
    chunk_duration: float,
    overlap: float,
    convert_to_mono: bool,
) -> bool:
    """Process a single audio file with chunking support."""

    print(f"\n{'='*70}")
    print(f"Processing: {audio_path.name}")
    print(f"{'='*70}")

    # Setup logger for this file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"sam_audio_{audio_path.stem}_{timestamp}.log"
    logger = setup_logger(log_file)

    logger.info("="*80)
    logger.info(f"Processing: {audio_path.name}")
    logger.info(f"Description: {description}")
    logger.info(f"Device: {device}")
    logger.info(f"Memory fraction: {memory_fraction}")
    logger.info(f"Rerank: {rerank}, Predict spans: {predict_spans}")
    logger.info(f"Chunk duration: {chunk_duration}s, Overlap: {overlap}s")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    print(f"Memory log: {log_file}")

    temp_files = []

    log_memory_stats(logger, "Initial state", device)

    try:
        # Convert to mono 16k if requested
        process_path = audio_path
        if convert_to_mono:
            print("[Step 1] Converting to mono 16kHz...")
            logger.info("Converting to mono 16kHz...")
            process_path = convert_to_mono_16k(audio_path)
            temp_files.append(process_path)
            print(f"✓ Converted: {process_path}")
            logger.info(f"Converted: {process_path}")
            log_memory_stats(logger, "After conversion", device)
        else:
            print("[Step 1] Skipping conversion (using original file)")
            logger.info("Skipping conversion (using original file)")

        # Check audio duration
        duration = get_audio_duration(process_path)
        print(f"[Step 2] Audio duration: {duration:.2f} seconds")
        logger.info(f"Audio duration: {duration:.2f} seconds")

        # Determine if chunking is needed
        use_chunking = duration > chunk_duration
        if use_chunking:
            print(f"[Step 3] Using chunking (chunk size: {chunk_duration}s, overlap: {overlap}s)")
            logger.info(f"Using chunking (chunk size: {chunk_duration}s, overlap: {overlap}s)")
        else:
            print("[Step 3] Processing entire file (no chunking needed)")
            logger.info("Processing entire file (no chunking needed)")

        if use_chunking:
            # Count chunks without loading into memory
            num_chunks = count_chunks(process_path, chunk_duration, overlap)
            print(f"✓ Will process {num_chunks} chunks (streaming mode - low memory)")
            logger.info(f"Will process {num_chunks} chunks (streaming mode)")
            log_memory_stats(logger, "Before chunk processing", device)

            target_chunks = []
            residual_chunks = []

            # CRITICAL: Use model's OUTPUT sample rate, not input sample rate
            output_sample_rate = int(processor.audio_sampling_rate)
            logger.info(f"Model output sample rate: {output_sample_rate} Hz")

            # Stream chunks one at a time using generator
            for i, (start, end, chunk_data, sr) in enumerate(chunk_audio_generator(process_path, chunk_duration, overlap), 1):

                print(f"  → Processing chunk {i}/{num_chunks} ({start:.1f}s - {end:.1f}s)...", end=" ")
                logger.info(f"Processing chunk {i}/{num_chunks} ({start:.1f}s - {end:.1f}s)")

                try:
                    target, residual = process_chunk(
                        chunk_data, sr, description,
                        model, processor, device,
                        rerank, predict_spans,
                        logger=logger,
                        chunk_idx=i
                    )

                    target_chunks.append(target)
                    residual_chunks.append(residual)
                    print("✓")

                except Exception as e:
                    print(f"✗ Failed")
                    logger.error(f"Chunk {i} failed: {e}", exc_info=True)
                    raise

                finally:
                    # CRITICAL: Explicitly delete chunk_data to free CPU memory
                    del chunk_data
                    # Aggressive CPU memory cleanup after each chunk
                    gc.collect()

                log_memory_stats(logger, f"After chunk {i} cleanup", device)

                # Show memory status
                if device == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"    GPU Memory: {allocated:.2f}GB allocated")

            # Merge chunks
            print(f"[Step 4] Merging {num_chunks} chunks...")
            logger.info(f"Merging {num_chunks} chunks...")
            log_memory_stats(logger, "Before merging", device)

            # Use model output sample rate for merging
            target_audio = merge_chunks(target_chunks, overlap, output_sample_rate)
            log_memory_stats(logger, "After merging targets", device)

            residual_audio = merge_chunks(residual_chunks, overlap, output_sample_rate)
            log_memory_stats(logger, "After merging residuals", device)

            # Set sample_rate for saving
            sample_rate = output_sample_rate

            # CRITICAL: Delete chunk lists immediately after merging
            del target_chunks, residual_chunks
            gc.collect()
            log_memory_stats(logger, "After deleting chunk lists", device)

            print("✓ Chunks merged")
            logger.info("Chunks merged successfully")

        else:
            # Process entire file
            print("[Step 4] Running inference...")
            logger.info("Running inference on entire file...")
            log_memory_stats(logger, "Before batch load", device)

            batch = processor(audios=[str(process_path)], descriptions=[description]).to(device)
            log_memory_stats(logger, "After batch load", device)

            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                result = model.separate(
                    batch,
                    predict_spans=predict_spans,
                    reranking_candidates=max(1, int(rerank)),
                )

            log_memory_stats(logger, "After inference", device)

            sample_rate = int(processor.audio_sampling_rate)

            # CRITICAL FIX: result.target/residual are lists, extract first element
            target_tensor = result.target[0] if isinstance(result.target, list) else result.target
            residual_tensor = result.residual[0] if isinstance(result.residual, list) else result.residual

            target_audio = target_tensor.squeeze().cpu().numpy().copy()
            residual_audio = residual_tensor.squeeze().cpu().numpy().copy()

            # Clean up immediately
            del batch, result, target_tensor, residual_tensor
            aggressive_cleanup()

            print("✓ Inference complete")
            logger.info("Inference complete")

        # Save outputs with unique names
        print("[Step 5] Saving results...")
        logger.info("Saving results...")
        log_memory_stats(logger, "Before saving", device)

        output_dir.mkdir(parents=True, exist_ok=True)

        stem = audio_path.stem
        target_path = get_unique_output_path(output_dir, f"{stem}_target")
        residual_path = get_unique_output_path(output_dir, f"{stem}_residual")

        sf.write(target_path, target_audio, sample_rate, subtype='PCM_16')
        sf.write(residual_path, residual_audio, sample_rate, subtype='PCM_16')

        print(f"✓ Target:   {target_path.name}")
        print(f"✓ Residual: {residual_path.name}")
        print(f"✓ Sample rate: {sample_rate} Hz")

        logger.info(f"Saved target: {target_path}")
        logger.info(f"Saved residual: {residual_path}")

        # CRITICAL: Delete audio arrays immediately after saving
        del target_audio, residual_audio
        gc.collect()
        log_memory_stats(logger, "After saving and cleanup", device)

        logger.info("Processing completed successfully")
        logger.info("="*80)

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        logger.error(f"ERROR: {e}", exc_info=True)
        log_memory_stats(logger, "At error", device)
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass


def batch_process(
    input_dir: Path,
    description: str,
    output_dir: Path,
    model_dir: Path,
    memory_fraction: float,
    rerank: int,
    predict_spans: bool,
    chunk_duration: float,
    overlap: float,
    convert_to_mono: bool,
):
    """Process all audio files in input directory."""

    # CRITICAL FIX: Prevent directory collision that causes infinite loops
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()

    if input_resolved == output_resolved:
        print("\n" + "="*70)
        print("ERROR: Input and output directories CANNOT be the same!")
        print("="*70)
        print(f"Input:  {input_resolved}")
        print(f"Output: {output_resolved}")
        print("\nThis would cause an infinite loop:")
        print("  1. Process file.wav → file_target.wav, file_residual.wav")
        print("  2. Detect file_target.wav as new input")
        print("  3. Process file_target.wav → file_target_target.wav, ...")
        print("  4. System crashes from infinite loop!")
        print("\nPlease choose different directories.")
        print("="*70 + "\n")
        sys.exit(1)

    # Warning if output is a subdirectory of input or vice versa
    try:
        if output_resolved.is_relative_to(input_resolved) or input_resolved.is_relative_to(output_resolved):
            print("\n" + "="*70)
            print("WARNING: Input and output directories overlap!")
            print("="*70)
            print(f"Input:  {input_resolved}")
            print(f"Output: {output_resolved}")
            print("\nThis may cause unexpected behavior.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ('y', 'yes'):
                print("Aborted.")
                sys.exit(0)
            print("="*70 + "\n")
    except AttributeError:
        # is_relative_to() not available in older Python versions
        pass

    # Find audio files
    audio_files = find_audio_files(input_dir)

    if not audio_files:
        print(f"\nNo audio files found in {input_dir}")
        print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return

    # CRITICAL FIX: Enforce maximum file limit to prevent runaway processing
    if len(audio_files) > MAX_FILES_PER_SESSION:
        print("\n" + "="*70)
        print(f"WARNING: Too many files detected ({len(audio_files)} files)")
        print("="*70)
        print(f"For safety, this script will only process up to {MAX_FILES_PER_SESSION} files per session.")
        print("This prevents infinite loops if output is being fed back as input.")
        print("\nOptions:")
        print(f"  1. Process first {MAX_FILES_PER_SESSION} files")
        print("  2. Cancel and check your input/output directory settings")
        response = input(f"\nProcess first {MAX_FILES_PER_SESSION} files? (y/N): ").strip().lower()
        if response not in ('y', 'yes'):
            print("Aborted.")
            return
        audio_files = audio_files[:MAX_FILES_PER_SESSION]
        print(f"Processing first {len(audio_files)} files...")

    print(f"\nFound {len(audio_files)} audio file(s) to process:")
    for i, f in enumerate(audio_files, 1):
        print(f"  {i}. {f.name}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)

    print(f"\nDevice: {device}")
    print(f"Memory fraction: {memory_fraction}")
    print(f"Description: '{description}'")
    print(f"Convert to mono 16kHz: {convert_to_mono}")
    print(f"Memory logs will be saved to: {LOG_DIR}")

    # Load model once
    print(f"\nLoading model from {model_dir}...")
    model = SAMAudio.from_pretrained(str(model_dir)).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(str(model_dir))
    print(f"✓ Model loaded")

    # Apply memory-safe reranking patch
    patch_sam_audio_model(model)

    # Process each file
    success_count = 0
    fail_count = 0
    processed_files = set()  # Track processed files to prevent reprocessing

    try:
        for audio_file in audio_files:
            # CRITICAL FIX: Skip if already processed in this session
            file_id = audio_file.resolve()
            if file_id in processed_files:
                print(f"\n⊗ Skipping {audio_file.name} (already processed in this session)")
                continue

            processed_files.add(file_id)
            success = process_audio_file(
                audio_file,
                description,
                output_dir,
                model,
                processor,
                device,
                memory_fraction,
                rerank,
                predict_spans,
                chunk_duration,
                overlap,
                convert_to_mono,
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

            # Cleanup between files
            aggressive_cleanup()

    finally:
        # Cleanup model
        del model, processor
        aggressive_cleanup()

    # Summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"Output directory: {output_dir}")
    print(f"Memory logs: {LOG_DIR}")
    print(f"{'='*70}\n")

    if fail_count > 0:
        print(f"\nNote: Check memory logs in {LOG_DIR} for crash analysis")


def interactive_mode():
    """Run in interactive mode with user prompts."""
    print("\n" + "="*70)
    print("SAM-Audio Interactive Batch CLI")
    print("="*70 + "\n")

    # Load saved config
    config = load_config()

    # Get user inputs
    print("Enter parameters (press Enter to use default):\n")

    input_dir = get_user_input(
        "Input directory (containing audio files)",
        config["input_dir"],
        str
    )

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Creating input directory: {input_path}")
        input_path.mkdir(parents=True, exist_ok=True)

    description = get_user_input(
        "Description (what to KEEP, e.g., 'speech', 'music')",
        config["last_description"],
        str
    )

    output_dir = get_user_input(
        "Output directory",
        config["output_dir"],
        str
    )

    model_dir = get_user_input(
        "Model directory",
        config["model_dir"],
        str
    )

    # Conversion option
    convert_input = get_user_input(
        "Convert to mono 16kHz before processing? (y/n)",
        "y" if config["convert_to_mono16k"] else "n",
        str
    )
    convert_to_mono = convert_input.lower() in ('y', 'yes')

    # Advanced options
    print("\nAdvanced options:")

    chunk_duration = get_user_input(
        "Chunk duration (seconds, 0=no chunking)",
        config["chunk_duration"],
        int
    )

    overlap = get_user_input(
        "Chunk overlap (seconds)",
        config["overlap"],
        float
    )

    memory_fraction = get_user_input(
        "GPU memory fraction (0.0-1.0)",
        config["memory_fraction"],
        float
    )

    rerank = get_user_input(
        "Reranking candidates (1-8)",
        config["rerank"],
        int
    )

    predict_spans_input = get_user_input(
        "Predict spans? (y/n)",
        "y" if config["predict_spans"] else "n",
        str
    )
    predict_spans = predict_spans_input.lower() in ('y', 'yes')

    # Save preferences
    config["input_dir"] = input_dir
    config["last_description"] = description
    config["output_dir"] = output_dir
    config["model_dir"] = model_dir
    config["convert_to_mono16k"] = convert_to_mono
    config["chunk_duration"] = chunk_duration
    config["overlap"] = overlap
    config["memory_fraction"] = memory_fraction
    config["rerank"] = rerank
    config["predict_spans"] = predict_spans

    save_config(config)

    # Process
    batch_process(
        Path(input_dir),
        description,
        Path(output_dir),
        Path(model_dir),
        memory_fraction,
        rerank,
        predict_spans,
        chunk_duration,
        overlap,
        convert_to_mono,
    )

    # Ask if user wants to process more
    again = input("\nProcess more files? (y/n): ").strip().lower()
    if again in ('y', 'yes'):
        interactive_mode()


def main():
    parser = argparse.ArgumentParser(
        description="SAM-Audio Interactive Batch CLI with Chunking Support"
    )
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults"
    )

    args = parser.parse_args()

    if args.reset_config:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        print("Configuration reset to defaults")
        return

    interactive_mode()


if __name__ == "__main__":
    main()
