================================================================================
WSL2 CRASH FIX - QUICK SUMMARY (2026-01-11)
================================================================================

PROBLEM:
--------
WSL2 crashed ALL sessions when running SAM-Audio script. Both the Python
session and Claude session were killed.

ROOT CAUSE:
-----------
- .wslconfig had swap configured (16GB) but swap file was NEVER CREATED
- WSL2 was running with 0GB swap despite configuration
- When system RAM exhausted → entire WSL2 VM crashed
- Script only monitored GPU memory, not system RAM

FIX APPLIED:
------------
Modified C:\Users\paul\.wslconfig:
1. Reduced memory: 28GB → 20GB (leave RAM for Windows to manage swap)
2. Changed swap path from %USERPROFILE% to absolute path:
   swapFile=C:\\Users\\paul\\AppData\\Local\\Temp\\wsl-swap.vhdx

VERIFICATION (Run these after WSL restart):
-------------------------------------------
1. Quick check:
   bash verify_wsl_fix.sh

2. Manual verification:
   swapon --show        # Should show swap device (not empty)
   free -h              # Should show ~16GB swap (not 0B)

3. Before re-running script, monitor in separate terminal:
   watch -n 1 'free -h'

FILES CREATED:
--------------
- WSL2_CRASH_DIAGNOSIS.md  - Full technical analysis & troubleshooting
- verify_wsl_fix.sh        - Quick verification script
- README_WSL_CRASH.txt     - This file (quick reference)

NEXT STEPS:
-----------
1. Restart WSL2 from PowerShell: wsl --shutdown
2. Wait 10 seconds
3. Restart WSL and run: bash verify_wsl_fix.sh
4. If swap is working, re-run your SAM-Audio command
5. Monitor memory in separate terminal

IMPORTANT:
----------
If swap STILL doesn't work after restart, see WSL2_CRASH_DIAGNOSIS.md
section "Fallback Solutions" for manual swap file creation steps.

================================================================================
