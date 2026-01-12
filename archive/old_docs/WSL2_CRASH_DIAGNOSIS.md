# WSL2 Crash Issue - Diagnosis & Resolution

**Date:** 2026-01-11
**Issue:** WSL2 crashes all Ubuntu sessions when running `run_sam_audio_ultrarobust.py`
**Status:** AWAITING VERIFICATION after WSL restart

---

## Problem Summary

When running the SAM-Audio script with:
```bash
python run_sam_audio_ultrarobust.py --file mnt/f/hf-home/audio_input/Campbell_Audio.wav --desc "young boy speaking, remove insect cicada noise" --verbose
```

**All WSL2 sessions crashed**, including both the session running the script and the Claude session.

---

## Root Cause Analysis

### What We Found

1. **System Evidence of Crash:**
   - `dmesg` showed journal corruption: `system.journal corrupted or uncleanly shut down`
   - This indicates an unclean shutdown/crash of the entire WSL2 VM

2. **CRITICAL ISSUE: Swap Not Working**
   - `.wslconfig` file exists at `C:\Users\paul\.wslconfig` with swap configured
   - BUT: `swapon --show` returned **empty** - no swap is actually mounted
   - BUT: `free -h` showed **0B swap** despite 16GB configured
   - Swap file does not exist at expected location: `C:\Users\paul\AppData\Local\Temp\wsl-swap.vhdx`

3. **Memory Configuration Status:**
   - `.wslconfig` memory limit (28GB) **IS working** - WSL has 27GB available
   - `.wslconfig` swap (16GB) **IS NOT working** - no swap file created
   - Username mapping: WSL user=`longboardfella`, Windows user=`paul`

4. **Why This Causes Crashes:**
   - SAM-Audio model requires 6-10GB+ **system RAM** (not just GPU memory)
   - Script has excellent GPU memory management but doesn't monitor system RAM
   - With 0GB swap, when system RAM fills up → WSL2 VM crashes entirely
   - This kills ALL sessions, not just the Python process

---

## Solution Applied

### Changes Made to `.wslconfig`

**Location:** `C:\Users\paul\.wslconfig` (Windows side)

**Changes:**
1. Reduced memory from `28GB` to `20GB` (leave more RAM for Windows to manage swap)
2. Changed swap file path from variable to absolute path:
   - OLD: `swapFile=%USERPROFILE%\\AppData\\Local\\Temp\\wsl-swap.vhdx`
   - NEW: `swapFile=C:\\Users\\paul\\AppData\\Local\\Temp\\wsl-swap.vhdx`

**New Configuration:**
```ini
[wsl2]
# Memory allocation
memory=20GB              # Reduced from 28GB - leave more RAM for Windows
swap=16GB                # Increase swap space for safety
swapFile=C:\\Users\\paul\\AppData\\Local\\Temp\\wsl-swap.vhdx  # Absolute path

# Processor allocation
processors=8

# Network
localhostForwarding=true

# GPU and memory settings
nestedVirtualization=true
pageReporting=true
guiApplications=true
```

**Action Required:**
WSL2 must be restarted for changes to take effect:
```powershell
# Run in PowerShell:
wsl --shutdown
# Wait 10 seconds, then restart WSL
```

---

## Verification Steps (Run After WSL Restart)

After restarting WSL2, run these commands to verify the fix:

```bash
# 1. Check if swap is now mounted
swapon --show
# EXPECTED: Should show a swap device (e.g., /dev/sdX or swap file)

# 2. Check memory allocation
free -h
# EXPECTED: Should show ~16GB swap in the "Swap:" line

# 3. Verify swap file exists on Windows side
ls -lh /mnt/c/Users/paul/AppData/Local/Temp/wsl-swap.vhdx
# EXPECTED: Should show a ~16GB file

# 4. Check total available RAM
echo "System RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Swap space: $(free -h | grep Swap | awk '{print $2}')"
# EXPECTED: ~19-20GB RAM, ~16GB swap
```

---

## Additional Monitoring Recommendations

### 1. Add System RAM Monitoring to Script

The `run_sam_audio_ultrarobust.py` script currently only monitors GPU memory. Consider adding:

```python
import psutil

def get_system_memory_stats():
    """Get current system RAM statistics in GB."""
    mem = psutil.virtual_memory()
    return {
        "total": f"{mem.total / (1024**3):.2f}GB",
        "available": f"{mem.available / (1024**3):.2f}GB",
        "used": f"{mem.used / (1024**3):.2f}GB",
        "percent": f"{mem.percent}%"
    }

def log_system_memory(stage: str):
    """Log current system memory state."""
    stats = get_system_memory_stats()
    print(f"[{stage}] System RAM: {stats['used']} used, "
          f"{stats['available']} available, {stats['percent']} usage")
```

### 2. Check for psutil Installation

```bash
pip list | grep psutil
# If not installed: pip install psutil
```

### 3. Monitor During Script Execution

```bash
# In a separate terminal, monitor memory usage:
watch -n 1 'free -h && echo "---" && swapon --show'
```

---

## Fallback Solutions (If Issue Persists)

### If Swap Still Doesn't Work After Restart:

**Option A: Pre-create swap file manually** (PowerShell as Administrator):
```powershell
$swapPath = "$env:USERPROFILE\AppData\Local\Temp\wsl-swap.vhdx"
New-Item -Path $swapPath -ItemType File -Force
# Set size to 16GB (17179869184 bytes)
fsutil file createnew $swapPath 17179869184
wsl --shutdown
```

**Option B: Use alternative swap location:**
```ini
swapFile=C:\\wsl-swap.vhdx
```

**Option C: Check Windows Event Viewer for errors:**
```powershell
Get-WinEvent -LogName Application -MaxEvents 100 | Where-Object {$_.Message -match "swap|vhdx|WSL"}
```

---

## Related Files

- **Script:** `/home/longboardfella/sam-audio/run_sam_audio_ultrarobust.py`
- **Config:** `C:\Users\paul\.wslconfig` (Windows side)
- **Swap file:** `C:\Users\paul\AppData\Local\Temp\wsl-swap.vhdx` (should exist after restart)

---

## Key Insights

1. **GPU memory ≠ System RAM:** The script manages GPU memory well but crashes were due to system RAM exhaustion
2. **No swap = Hard crash:** Without swap, Linux OOM killer crashes entire WSL2 VM
3. **`.wslconfig` variable expansion issues:** `%USERPROFILE%` variable may not expand properly in swap path
4. **Memory allocation balance:** Allocating too much to WSL2 (28GB) may prevent Windows from managing swap properly

---

## Status After Fix

- [ ] WSL2 restarted
- [ ] Swap verified with `swapon --show`
- [ ] Swap file exists on disk
- [ ] Script tested successfully without crashes
- [ ] System RAM monitoring added (optional)

**Next test:** Re-run the same command that caused the crash and monitor with `watch -n 1 'free -h'` in a separate terminal.
