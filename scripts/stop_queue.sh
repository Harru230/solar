#!/usr/bin/env bash
# scripts/stop_queue.sh - Request a clean stop of the GPU queue after the current epoch finishes and saves its checkpoint.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

STOP_FLAG="work/STOP_QUEUE"
mkdir -p work
touch "$STOP_FLAG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Created $STOP_FLAG."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] The active training run will finish the current epoch, save checkpoint_fold<k>_last.pt atomically, and exit cleanly."

if [[ "${1:-}" == "--nowait" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --nowait specified; returning immediately while queue finishes current epoch."
    exit 0
fi

if ! tmux has-session -t queue 2>/dev/null; then
    rm -f "$STOP_FLAG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No tmux session 'queue' is running. Nothing to wait for."
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for current epoch to finish and tmux session 'queue' to exit..."
while tmux has-session -t queue 2>/dev/null; do
    latest_log=$(ls -t work/queue_logs/*.log 2>/dev/null | head -1 || true)
    if [[ -n "$latest_log" ]]; then
        last_line=$(tail -n 1 "$latest_log" 2>/dev/null || true)
        echo "  [$(date '+%H:%M:%S')] Latest log ($latest_log): $last_line"
    fi
    sleep 10
done

rm -f "$STOP_FLAG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue stopped cleanly at epoch boundary. Safe to shut down PC."
