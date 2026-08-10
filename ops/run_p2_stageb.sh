#!/bin/bash
# Phase 2 Stage B: dual tree/flat probe at each budget. Waits for the
# side-probe GUARD to finish (it holds SearchEval.exe), rebuilds with
# the --flat-compare driver, then one match per budget with the flat
# evaluator pairing every non-forced play decision.
cd "$(dirname "$0")/.."
LOG=logs/p2_stageb.log
cat /proc/$$/winpid > logs/p2_stageb.pid
echo "STAGEB waiting for guard $(date)" >> "$LOG"
until grep -q "GUARD done" logs/probe_guard.log 2>/dev/null; do sleep 300; done
echo "STAGEB rebuild $(date)" >> "$LOG"
cmake --build build --config Release --target SearchEval >> "$LOG" 2>&1 \
  || { echo "STAGEB BUILD FAILED $(date)" >> "$LOG"; exit 1; }
for B in 200 400 800; do
  echo "STAGEB budget $B start $(date)" >> "$LOG"
  build/Release/SearchEval --tree-selfplay \
    --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --iterations $B --k 24 --matches 1 --seed $((212000000 + B)) --cuda \
    --flat-compare expert_data/p2/stageb_$B.csv --compare-k 64 \
    --record-out expert_data/p2/stageb_$B.hvt \
    > logs/p2_stageb_$B.log 2>&1 \
    || { echo "STAGEB budget $B FAILED $(date)" >> "$LOG"; exit 1; }
  python validate_p2_records.py expert_data/p2/stageb_$B.hvt --iters $B >> "$LOG" 2>&1
done
echo "STAGEB_COMPLETE $(date)" >> "$LOG"
