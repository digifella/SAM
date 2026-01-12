#!/bin/bash
# Memory monitoring script for SAM-Audio processing
# Logs both system RAM and GPU memory every 2 seconds

LOG_FILE="memory_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "Starting memory monitor - logging to $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""
echo "Timestamp,RAM_Used_GB,RAM_Avail_GB,RAM_Percent,Swap_Used_GB,GPU_Used_MB,GPU_Free_MB" > "$LOG_FILE"

while true; do
    TIMESTAMP=$(date +%H:%M:%S)

    # Get RAM stats
    RAM_STATS=$(free -g | grep Mem)
    RAM_USED=$(echo $RAM_STATS | awk '{print $3}')
    RAM_AVAIL=$(echo $RAM_STATS | awk '{print $7}')
    RAM_TOTAL=$(echo $RAM_STATS | awk '{print $2}')
    RAM_PERCENT=$(echo "scale=1; $RAM_USED * 100 / $RAM_TOTAL" | bc)

    # Get Swap stats
    SWAP_STATS=$(free -g | grep Swap)
    SWAP_USED=$(echo $SWAP_STATS | awk '{print $3}')

    # Get GPU stats
    if command -v nvidia-smi &> /dev/null; then
        GPU_STATS=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits)
        GPU_USED=$(echo $GPU_STATS | cut -d',' -f1 | xargs)
        GPU_FREE=$(echo $GPU_STATS | cut -d',' -f2 | xargs)
    else
        GPU_USED="N/A"
        GPU_FREE="N/A"
    fi

    # Display to console
    printf "\r[%s] RAM: %sGB/%sGB (%s%%) | Swap: %sGB | GPU: %sMB used, %sMB free" \
        "$TIMESTAMP" "$RAM_USED" "$RAM_TOTAL" "$RAM_PERCENT" "$SWAP_USED" "$GPU_USED" "$GPU_FREE"

    # Log to file
    echo "$TIMESTAMP,$RAM_USED,$RAM_AVAIL,$RAM_PERCENT,$SWAP_USED,$GPU_USED,$GPU_FREE" >> "$LOG_FILE"

    sleep 2
done
