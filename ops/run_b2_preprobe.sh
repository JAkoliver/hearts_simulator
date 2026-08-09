#!/bin/bash
# Round-2 Phase B2 defense PRE-PROBE (prereg: non-binding, ORDERING
# only, 16 paired seeds vs the frozen SEL probe) - picks the best <=2
# drift-screen passers for full gating.
#
# SEED BASE 190000000 - deliberately FRESH, NOT the defense-gate block
# 720260806: selecting candidates on gate seeds would correlate the
# selection with gate noise and inflate pass odds. Audited disjoint
# from all prior blocks (20-51M, 70-92M, 100-115M, 130-155M, 160-179M
# generation, 520/620M, 720-722M).
#
# 16 paired matches = 2 shards x 8. Base arm runs once; candidate arms
# on identical seeds (CRN). Same CSV surface as ops/run_defense_gate.sh.
# Usage: bash ops/run_b2_preprobe.sh <arm_name> <defender_trace.pt>
#   (run once with "base hearts_ai_match.pt", then per candidate)
cd "$(dirname "$0")/.."
NAME=${1:?arm name required}
DEF=${2:?defender trace required}
SHARDS=2
PER=8
EXE=build/Release/SearchEval
OUT=equity_data/exploiter_r2/preprobe
LOG=logs/b2_preprobe.log
mkdir -p "$OUT" logs
echo "PREPROBE arm=$NAME def=$(md5sum "$DEF" | cut -c1-8) probe=$(md5sum hearts_ai_search_match.pt | cut -c1-8) $(date)" >> "$LOG"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --opponent-model "$DEF" --shooter sel --pass-search \
    --k 64 --matches "$PER" --seed $((190000000 + i * 1000000)) --cuda \
    --out "$OUT/${NAME}_${i}.csv" \
    --tricks-out "$OUT/${NAME}_${i}.tricks.csv" \
    > "logs/b2_preprobe_${NAME}_${i}.log" 2>&1 &
  pids+=($!)
done
sleep 3
powershell -NoProfile -Command \
  "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
  > /dev/null 2>&1
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "PREPROBE arm=$NAME done rc=$rc $(date)" >> "$LOG"
exit $rc
