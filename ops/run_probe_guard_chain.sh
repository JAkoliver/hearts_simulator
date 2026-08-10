#!/bin/bash
# Waits for the Phase 2 pace probe to free the GPU, then runs the
# side-probe guard (registered in the Phase 2 prereg).
cd "$(dirname "$0")/.."
LOG=logs/probe_guard.log
cat /proc/$$/winpid > logs/probe_guard.pid
echo "GUARD waiting for pace probe $(date)" >> "$LOG"
until grep -qE "P2PACE_COMPLETE|FAILED" logs/p2_paceprobe.log 2>/dev/null; do
  sleep 300
done
echo "GUARD start $(date)" >> "$LOG"
HEARTS_HEADROOM=${HEARTS_HEADROOM:-0.25} PYTHONUNBUFFERED=1 \
  python -u run_probe_guard.py >> "$LOG" 2>&1
echo "GUARD done rc=$? $(date)" >> "$LOG"
