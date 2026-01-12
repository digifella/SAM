# SAM-Audio WSL2 Troubleshooting Guide

## Quick Start: Use Ultra-Robust Script

```bash
# Source environment setup
source setup_robust_env.sh

# Run with your audio file
python run_sam_audio_ultrarobust.py --file audio.wav --desc "your description" --verbose
```

## Common Issues and Solutions

### 1. WSL2 Crashes / System Becomes Unresponsive

**Symptoms:**
- WSL2 terminal freezes
- Windows becomes sluggish
- Need to restart WSL2 or entire system

**Root Causes:**
- WSL2 consuming all available system memory
- No memory limits configured

**Solution:**
1. Copy `.wslconfig` to Windows user directory:
   ```powershell
   # In PowerShell, copy the config file
   Copy-Item \\wsl$\Ubuntu\home\longboardfella\.wslconfig C:\Users\longboardfella\.wslconfig
   ```

2. Restart WSL2:
   ```powershell
   # In PowerShell (Admin)
   wsl --shutdown
   ```

3. Reopen WSL2 terminal and verify:
   ```bash
   free -h
   # Should show ~28GB memory limit
   ```

### 2. CUDA Out of Memory (OOM) Errors

**Symptoms:**
- Error: "CUDA out of memory"
- Script crashes during inference

**Solutions (in order of preference):**

**A. Use the ultra-robust script (handles OOM automatically):**
```bash
python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech"
```

**B. Reduce memory fraction:**
```bash
python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech" --memory_fraction 0.5
```

**C. Disable reranking and spans (simplest mode):**
```bash
python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech" --rerank 1
```

**D. Clear GPU memory before running:**
```bash
# Kill any Python processes using GPU
pkill -9 python

# Verify GPU is clear
nvidia-smi
```

### 3. Gradual Memory Leaks / Multiple Runs Failing

**Symptoms:**
- First run works, subsequent runs fail
- Memory usage grows with each run
- Need to restart terminal

**Root Cause:**
- Incomplete memory cleanup between runs
- PyTorch caching memory

**Solution:**
The ultra-robust script handles this automatically. If using other scripts:

```bash
# Add aggressive cleanup between runs
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# Or restart Python each time (simplest)
for file in audio1.wav audio2.wav; do
    python run_sam_audio_ultrarobust.py --file "$file" --desc "speech"
done
```

### 4. FFmpeg Conversion Issues

**Symptoms:**
- Error: "ffmpeg binary not found"
- Conversion fails

**Solution:**
```bash
# Install ffmpeg
sudo apt update
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

### 5. Model Loading Fails

**Symptoms:**
- Error: "Model not found"
- Loading hangs indefinitely

**Solution:**
```bash
# Check model directory exists
ls -lh ~/models/sam-audio-large-tv/

# If missing, you need to download the model
# Verify MODEL_DIR environment variable
echo $SAM_MODEL_DIR

# Or specify explicitly
python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech" \
    --model_dir ~/models/sam-audio-large-tv
```

### 6. Performance is Slow

**Symptoms:**
- Processing takes much longer than expected
- GPU utilization is low

**Possible Causes & Solutions:**

**A. CPU bottleneck during data loading:**
```bash
# Check if CPU is bottlenecked
htop  # Watch during processing

# If CPU at 100%, reduce parallel data loading in processor
```

**B. Excessive memory cleanup:**
The ultra-robust script is conservative. For faster processing on stable systems:
```bash
# Use higher memory fraction
python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech" \
    --memory_fraction 0.85
```

**C. WSL2 virtualization overhead:**
- Expected: ~20-30% slower than native Linux
- Use Windows Task Manager to check if Windows is memory-constrained

### 7. Debugging Memory Issues

**Enable verbose logging:**
```bash
python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech" --verbose
```

This shows memory usage at each stage:
```
[Initial] GPU Memory: 0.50GB allocated, 0.50GB reserved, 45.50GB free of 48.00GB
[After processor load] GPU Memory: 0.52GB allocated, ...
[After model load] GPU Memory: 5.20GB allocated, ...
[After inference] GPU Memory: 12.50GB allocated, ...
```

**Check system memory during processing:**
```bash
# In another terminal, watch memory usage
watch -n 1 'free -h && nvidia-smi --query-gpu=memory.used,memory.free --format=csv'
```

**Check for memory fragmentation:**
```bash
# After several runs, check fragmentation
nvidia-smi -q -d MEMORY
```

## Memory Budget Guidelines

For 48GB RTX 8000 on WSL2:

| Audio Duration | Rerank | Spans | Est. Peak Memory | Safety Margin | Fraction |
|----------------|--------|-------|------------------|---------------|----------|
| < 30 seconds   | 1      | No    | ~6-8 GB          | Safe          | 0.7      |
| 30-60 seconds  | 1      | No    | ~8-12 GB         | Safe          | 0.7      |
| 60-120 seconds | 1      | No    | ~12-18 GB        | Moderate      | 0.6      |
| > 120 seconds  | 1      | No    | ~18-25 GB        | Use chunking  | 0.5      |
| Any + rerank>1 | 2-4    | No    | Add 50-100%      | Risky         | 0.5      |
| Any + spans    | Any    | Yes   | Add 30-50%       | Very Risky    | 0.5      |

## Advanced: PyTorch Memory Allocator Settings

The ultra-robust script sets these automatically:

```bash
# Reduce fragmentation (set before import torch)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
```

Settings explanation:
- `max_split_size_mb:512` - Prevents large block fragmentation
- `expandable_segments:True` - Allows dynamic expansion (better for varied workloads)

Alternative for very constrained memory:
```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"
```

## When All Else Fails

1. **Complete reset:**
   ```bash
   # Kill all Python processes
   pkill -9 python

   # Clear GPU memory
   nvidia-smi --gpu-reset

   # Restart WSL2
   # In PowerShell: wsl --shutdown
   ```

2. **Use CPU mode (slow but stable):**
   ```bash
   # Temporarily disable CUDA
   export CUDA_VISIBLE_DEVICES=""
   python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech"
   ```

3. **Process in chunks (for very long audio):**
   ```bash
   # Split audio first
   ffmpeg -i long_audio.wav -f segment -segment_time 30 -c copy chunk_%03d.wav

   # Process each chunk
   for chunk in chunk_*.wav; do
       python run_sam_audio_ultrarobust.py --file "$chunk" --desc "speech"
   done
   ```

## Monitoring Tools

**Real-time GPU monitoring:**
```bash
# Terminal 1: Run processing
python run_sam_audio_ultrarobust.py --file audio.wav --desc "speech" --verbose

# Terminal 2: Watch GPU
watch -n 0.5 nvidia-smi
```

**Log memory over time:**
```bash
# Save memory log
while true; do
    date +%H:%M:%S
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    sleep 1
done > memory_log.txt
```

## Getting Help

If issues persist:
1. Enable `--verbose` logging
2. Capture full error message
3. Note: GPU memory at crash time, audio duration, command used
4. Check `dmesg | tail -50` for kernel OOM messages
