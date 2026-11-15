#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
source .venv/bin/activate

# WSL2 GPU Stability exports
export CUDA_VISIBLE_DEVICES=0

QUEUE_FILE="work/queue.txt"
LOG_DIR="work/queue_logs"
STOP_FLAG="work/STOP_QUEUE"
mkdir -p "$LOG_DIR"
rm -f "$STOP_FLAG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting run_queue.sh inside tmux session 'queue'"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watching $QUEUE_FILE for jobs..."

# Find next unexecuted job
line_num=1
while [[ -f "${LOG_DIR}/${line_num}.log" ]] && grep -q "Finished at:" "${LOG_DIR}/${line_num}.log"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job #${line_num} already finished. Skipping."
    ((line_num++))
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming queue from Job #${line_num}."

while true; do
    if [[ -f "$STOP_FLAG" ]]; then
        rm -f "$STOP_FLAG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Clean stop flag ($STOP_FLAG) detected before Job #${line_num}. Exiting queue."
        exit 0
    fi

    if [[ ! -f "$QUEUE_FILE" ]]; then
        sleep 5
        continue
    fi

    total_lines=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo 0)
    if [[ "$line_num" -gt "$total_lines" ]]; then
        sleep 5
        continue
    fi

    cmd=$(sed -n "${line_num}p" "$QUEUE_FILE")
    trimmed=$(echo "$cmd" | xargs)

    if [[ -z "$trimmed" || "$trimmed" == \#* ]]; then
        ((line_num++))
        continue
    fi

    log_file="${LOG_DIR}/${line_num}.log"
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Job #${line_num}:"
    echo "Command: $cmd"
    echo "Log: $log_file"
    echo "=================================================================="

    start_ts=$(date +%s)
    {
        echo "=== Queue Job #${line_num} ==="
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Command:   $cmd"
        echo "==============================="
    } >> "$log_file"

    eval "$cmd" 2>&1 | tee -a "$log_file"
    ret=${PIPESTATUS[0]}

    if [[ $ret -eq 0 ]]; then
        {
            echo "==============================="
            echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Exit code:   $ret"
            echo "Elapsed:     $(( $(date +%s) - start_ts ))s"
        } | tee -a "$log_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job #${line_num} finished with exit code $ret."
        ((line_num++))
    elif [[ $ret -eq 130 ]]; then
        rm -f "$STOP_FLAG"
        {
            echo "==============================="
            echo "PAUSED at:   $(date '+%Y-%m-%d %H:%M:%S') (clean epoch-boundary stop)"
            echo "Exit code:   $ret"
            echo "Elapsed:     $(( $(date +%s) - start_ts ))s"
        } | tee -a "$log_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job #${line_num} paused cleanly at epoch boundary. Will resume on next run."
        exit 0
    else
        {
            echo "==============================="
            echo "FAILED at:   $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Exit code:   $ret"
            echo "Elapsed:     $(( $(date +%s) - start_ts ))s"
        } | tee -a "$log_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job #${line_num} FAILED with exit code $ret. Stopping queue."
        exit "$ret"
    fi
done
