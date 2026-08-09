#!/bin/bash
# Round-2 DEFENSE GATE, candidate arm only: the round-1 base arm is
# REUSED per prereg (deterministic on the frozen instruments -
# base_*.csv md5-identical across all three r1 runs, verified
# 2026-08-09) and copied in by the battery driver. Same frozen SEL
# probe, same seed block 720260806, 2 shards x 32 = 64 CRN matches.
# Usage: bash ops/run_r2_defense_gate.sh <tag> <cand_trace.pt>
cd "$(dirname "$0")/.."
TAG=${1:?tag required}
CAND=${2:?candidate trace required}
SHARDS=2
PER=32
EXE=build/Release/SearchEval
OUT=equity_data/exploiter_r2/gate_$TAG
LOG=logs/r2_defense_gate_$TAG.log
mkdir -p "$OUT" logs
echo "R2DEFGATE $TAG cand=$(md5sum "$CAND" | cut -c1-8) probe=$(md5sum hearts_ai_search_match.pt | cut -c1-8) $(date)" >> "$LOG"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --opponent-model "$CAND" --shooter sel --pass-search \
    --k 64 --matches "$PER" --seed $((720260806 + i * 1000000)) --cuda \
    --out "$OUT/cand_${i}.csv" \
    > "logs/r2_defgate_${TAG}_${i}.log" 2>&1 &
  pids+=($!)
done
sleep 3
powershell -NoProfile -Command \
  "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
  > /dev/null 2>&1
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "R2DEFGATE $TAG done rc=$rc $(date)" >> "$LOG"
exit $rc
