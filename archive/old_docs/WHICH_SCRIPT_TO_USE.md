# Which Script to Use - SAM-Audio Processing

After WSL crashes and multiple versions, here's the clear guide on which script to use.

## ✅ RECOMMENDED: run_sam_interactive.py

**This is the script you should use going forward.**

### Features:
- ✅ Interactive prompts for all settings
- ✅ Processes ALL files in input directory
- ✅ Saves your preferences between runs
- ✅ Batch processing with chunking support
- ✅ Auto-increments output filenames
- ✅ **Memory-safe reranking patch for WSL2** (FIXED the crash!)

### How to Use:

```bash
python run_sam_interactive.py
```

Then answer the prompts:
- Input directory (containing audio files): `./audio_input`
- Description (what to KEEP): `speech`
- Output directory: `./audio_output`
- Model directory: `~/models/sam-audio-large-tv`
- Convert to mono 16kHz: `y`
- Chunk duration: `60` (processes files in 60-second chunks)
- Overlap: `2.0` (seconds of overlap between chunks)
- GPU memory fraction: `0.7` (safe for WSL2)
- Reranking candidates: `1` or `2` (now safe with patch!)
- Predict spans: `n`

Your settings are saved to `~/.sam_audio_config.json` and will be the defaults next time.

### What Got Fixed:

The script now includes a **memory-safe reranking patch** that:
1. Moves waveforms to CPU after ODE generation
2. Adds `torch.cuda.synchronize()` before cleanup (critical for WSL2!)
3. Cleans up memory between heavy operations
4. Runs CLAP reranking on CPU to prevent GPU OOM

This fixes the crash you experienced on the "2nd pass of reranking" with your 28-second file.

## Other Scripts (Reference Only)

### run_sam_audio_patched.py
- Single-file processing with memory-safe reranking
- Use if you only want to process one file at a time with command-line args
- Same patch as interactive version

### run_sam_audio_ultrarobust.py
- Single-file processing WITHOUT the reranking patch
- Has fallback to simpler settings on OOM
- Use rerank=1 to avoid crashes

### run_sam_audio_batch*.py
- Various old batch processing attempts
- **DO NOT USE** - these don't have the memory-safe patch

### run_sam_audio.py
- Original simple script
- **DO NOT USE** for WSL2

## Summary

**Just use `run_sam_interactive.py`** - it has everything you need:
- Interactive prompts
- Batch processing
- Memory-safe reranking
- WSL2 stability fixes
- Saved preferences

Place your audio files in the input directory and run the script. It will process everything and save your preferences for next time.
