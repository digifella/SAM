#!/bin/bash
# Setup script for ultra-robust SAM-Audio environment on WSL2

echo "=========================================="
echo "SAM-Audio Ultra-Robust Environment Setup"
echo "=========================================="
echo ""

# Set PyTorch memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Reduce CUDA memory fragmentation
export CUDA_LAUNCH_BLOCKING=0  # Async execution (faster)

# Optional: Reduce PyTorch multithreading if you see CPU contention
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4

echo "Environment variables set:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo ""

# Check GPU availability
echo "Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Check system memory
echo "System memory:"
free -h | grep -E "Mem:|Swap:"
echo ""

# Check WSL2 configuration
echo "WSL2 Configuration Check:"
if [ -f "$HOME/.wslconfig" ]; then
    echo "  Found .wslconfig in home directory (needs to be copied to Windows)"
    echo "  Copy to: C:\\Users\\$(whoami)\\.wslconfig"
    echo "  Then run in PowerShell: wsl --shutdown"
    echo ""
else
    echo "  No .wslconfig found. Using WSL2 defaults."
    echo ""
fi

echo "=========================================="
echo "Setup complete!"
echo ""
echo "Usage examples:"
echo ""
echo "  # Basic usage (most stable):"
echo "  python run_sam_audio_ultrarobust.py --file audio.wav --desc 'speech'"
echo ""
echo "  # With verbose memory logging:"
echo "  python run_sam_audio_ultrarobust.py --file audio.wav --desc 'speech' --verbose"
echo ""
echo "  # Adjust memory fraction (0.5-0.9 range):"
echo "  python run_sam_audio_ultrarobust.py --file audio.wav --desc 'speech' --memory_fraction 0.6"
echo ""
echo "=========================================="
