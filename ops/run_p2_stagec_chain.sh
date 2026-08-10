#!/bin/bash
# Phase 2 Stage C chain (prereg rider, signed 2026-08-10):
#   1. rebuild SearchEval (tree equity in --deals driver)
#   2. STRENGTH SCREEN: tree-200 (Stage C teacher config) vs flat-64,
#      CRN 2x400 deals, seeds 213M block; band UB(tree-flat) <= +1.0
#   3. on PASS: launch the generation bank (2 x 170 matches, seeds
#      210,000,000 / 210,500,000, budget 200) - user pre-authorized.
#      On FAIL: halt before any generation spend.
# Markers: "P2C <event>" in logs/p2_stagec_chain.log
cd "$(dirname "$0")/.."
LOG=logs/p2_stagec_chain.log
mkdir -p logs expert_data/p2
cat /proc/$$/winpid > logs/p2_stagec_chain.pid
mark() { echo "P2C $* $(date)" >> "$LOG"; }

mark rebuild
cmake --build build --config Release --target SearchEval >> "$LOG" 2>&1 \
  || { mark "BUILD FAILED"; exit 1; }

EXE=build/Release/SearchEval
NEUT=legacy_v3_pass238/hearts_ai_grandmaster_v3_milestone7.pt
mark "strength screen start"
pids=()
for i in 0 1; do
  S=$((213000000 + i * 1000000))
  "$EXE" --search-model hearts_ai_search_match.pt --opponent-model "$NEUT" \
    --equity-model hearts_equity.pt --deals 400 --k 24 --pass-search \
    --tree --iterations 200 --seed $S --cuda \
    --out expert_data/p2/strength_tree_s$i.csv \
    > logs/p2_strength_tree_$i.log 2>&1 &
  pids+=($!)
  "$EXE" --search-model hearts_ai_search_match.pt --opponent-model "$NEUT" \
    --equity-model hearts_equity.pt --deals 400 --k 64 --pass-search \
    --seed $S --cuda \
    --out expert_data/p2/strength_flat_s$i.csv \
    > logs/p2_strength_flat_$i.log 2>&1 &
  pids+=($!)
done
sleep 5
powershell -NoProfile -Command \
  "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
  > /dev/null 2>&1
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -ne 0 ] && { mark "STRENGTH SHARD-FAIL"; exit 1; }

PYTHONUNBUFFERED=1 python -u analyze_p2_strength.py >> "$LOG" 2>&1
src=$?
if [ $src -ne 0 ]; then
  mark "STRENGTH HALT (band missed) - no generation"
  exit 1
fi
mark "STRENGTH PASS -> generation launch"

pids=()
for i in 0 1; do
  S=$((210000000 + i * 500000))
  "$EXE" --tree-selfplay \
    --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --iterations 200 --k 24 --matches 170 --seed $S --cuda \
    --record-out expert_data/p2/bank_s$i.hvt \
    > logs/p2_bank_$i.log 2>&1 &
  pids+=($!)
done
sleep 5
powershell -NoProfile -Command \
  "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
  > /dev/null 2>&1
mark "generation running (2 shards x 170 matches, seeds 210000000/210500000)"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -ne 0 ] && { mark "GENERATION SHARD-FAIL"; exit 1; }
for i in 0 1; do
  PYTHONUNBUFFERED=1 python -u validate_p2_records.py \
    expert_data/p2/bank_s$i.hvt --iters 200 >> "$LOG" 2>&1
done
mark "GENERATION COMPLETE"
