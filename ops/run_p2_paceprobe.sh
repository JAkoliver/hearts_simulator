#!/bin/bash
# Phase 2 Stage C pace probe (prereg): ~20 deals at iteration budgets
# {200, 400, 800}. WAITS for the lambda=0.05 side-probe trainer to
# free the GPU (R3TRIAL probe005 COMPLETE marker), then runs the three
# budgets sequentially at BelowNormal. Signal stats per budget via
# validate_p2_records.py feed the one-time budget choice.
cd "$(dirname "$0")/.."
LOG=logs/p2_paceprobe.log
mkdir -p expert_data/p2 logs
cat /proc/$$/winpid > logs/p2_paceprobe.pid
echo "P2PACE waiting for side-probe $(date)" >> "$LOG"
until grep -q "R3TRIAL probe005 \(COMPLETE\|HALT\)" logs/r3_trial_probe005.log; do
  sleep 120
done
echo "P2PACE start $(date)" >> "$LOG"
for B in 200 400 800; do
  echo "P2PACE budget $B start $(date)" >> "$LOG"
  build/Release/SearchEval --tree-selfplay \
    --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --iterations $B --k 24 --matches 3 --seed $((211000000 + B)) --cuda \
    --record-out expert_data/p2/pace_$B.hvt \
    > logs/p2_pace_$B.log 2>&1 || { echo "P2PACE budget $B FAILED $(date)" >> "$LOG"; exit 1; }
  sleep 2
  powershell -NoProfile -Command \
    "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
    > /dev/null 2>&1
  python validate_p2_records.py expert_data/p2/pace_$B.hvt --iters $B >> "$LOG" 2>&1
  echo "P2PACE budget $B done $(date)" >> "$LOG"
done
echo "P2PACE_COMPLETE $(date)" >> "$LOG"
