#!/bin/bash
# Quick verification script for WSL2 swap fix
# Run this after restarting WSL2 with: bash verify_wsl_fix.sh

echo "=========================================="
echo "WSL2 Swap Fix Verification"
echo "=========================================="
echo ""

# Check 1: Swap mounted
echo "1. Checking if swap is mounted..."
SWAP_OUTPUT=$(swapon --show)
if [ -z "$SWAP_OUTPUT" ]; then
    echo "   ❌ FAILED: No swap is mounted"
    echo "   Action: Check WSL2_CRASH_DIAGNOSIS.md for troubleshooting"
else
    echo "   ✅ SUCCESS: Swap is mounted"
    echo "$SWAP_OUTPUT"
fi
echo ""

# Check 2: Free memory
echo "2. Memory allocation:"
free -h
SWAP_SIZE=$(free -h | grep Swap | awk '{print $2}')
if [ "$SWAP_SIZE" = "0B" ]; then
    echo "   ❌ WARNING: Swap size is 0B"
else
    echo "   ✅ Swap: $SWAP_SIZE"
fi
echo ""

# Check 3: Swap file exists
echo "3. Checking swap file on Windows side..."
if [ -f /mnt/c/Users/paul/AppData/Local/Temp/wsl-swap.vhdx ]; then
    SIZE=$(ls -lh /mnt/c/Users/paul/AppData/Local/Temp/wsl-swap.vhdx | awk '{print $5}')
    echo "   ✅ SUCCESS: Swap file exists ($SIZE)"
else
    echo "   ❌ FAILED: Swap file does not exist"
    echo "   Expected: /mnt/c/Users/paul/AppData/Local/Temp/wsl-swap.vhdx"
fi
echo ""

# Check 4: Memory stats
echo "4. System summary:"
TOTAL_RAM=$(free -h | grep Mem | awk '{print $2}')
TOTAL_SWAP=$(free -h | grep Swap | awk '{print $2}')
echo "   Total RAM:  $TOTAL_RAM"
echo "   Total Swap: $TOTAL_SWAP"
echo ""

# Check 5: psutil for future monitoring
echo "5. Checking for psutil (optional system RAM monitoring)..."
if python3 -c "import psutil" 2>/dev/null; then
    echo "   ✅ psutil is installed"
else
    echo "   ⚠️  psutil not installed (optional)"
    echo "   To install: pip install psutil"
fi
echo ""

# Summary
echo "=========================================="
if [ -z "$SWAP_OUTPUT" ] || [ "$SWAP_SIZE" = "0B" ]; then
    echo "STATUS: ❌ SWAP NOT WORKING"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Read WSL2_CRASH_DIAGNOSIS.md for troubleshooting"
    echo "2. Verify .wslconfig changes were saved"
    echo "3. Try: wsl --shutdown (in PowerShell) and restart"
    echo "4. Consider manual swap file creation (see diagnosis doc)"
else
    echo "STATUS: ✅ SWAP CONFIGURED CORRECTLY"
    echo "=========================================="
    echo ""
    echo "You can now safely run your SAM-Audio script."
    echo "Monitor memory in real-time with:"
    echo "  watch -n 1 'free -h && echo --- && swapon --show'"
fi
echo ""
