#!/bin/bash
# Exploiter-league round-1 DEFENSE GATE (docs/exploiter_league_prereg.md
# + 2026-08-07 amendment): frozen SEL search-shooter probe (K=64 flat,
# pass search, hearts_ai_search_match.pt af8cc2bd) attacking a field of
# 3 defenders, run twice on IDENTICAL seeds (CRN-paired):
#   arm "base": defenders = hearts_ai_match.pt (baseline 8a89da90 trace)
#   arm "cand": defenders = the candidate's 556 match trace
# 64 paired seed-matches = 2 shards x 32, seed block base 720260806
# (fresh space per prereg). 2 shards, BelowNormal: gentler than the
# validated 3-shard Phase A/B load - the desktop stays usable.
#
# Usage: nohup bash ops/run_defense_gate.sh <cand_trace.pt> <tag> &
cd "$(dirname "$0")/.."
CAND=${1:?candidate trace required}
TAG=${2:-r1t1}
SHARDS=2
PER=32
EXE=build/Release/SearchEval
OUT=equity_data/exploiter_r1/gate_$TAG
LOG=logs/defense_gate_$TAG.log
mkdir -p "$OUT" logs
cat /proc/$$/winpid > logs/defgate_driver.pid
echo "DEFGATE_START $(date) cand=$(md5sum "$CAND" | cut -c1-8) base=$(md5sum hearts_ai_match.pt | cut -c1-8) probe=$(md5sum hearts_ai_search_match.pt | cut -c1-8)" >> "$LOG"

run_arm() {  # armname defender_trace
  local name=$1 def=$2
  echo "DEFGATE_ARM $name start $(date)" >> "$LOG"
  local pids=() i
  for i in $(seq 0 $((SHARDS - 1))); do
    "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
      --opponent-model "$def" --shooter sel --pass-search \
      --k 64 --matches "$PER" --seed $((720260806 + i * 1000000)) --cuda \
      --out "$OUT/${name}_${i}.csv" \
      --tricks-out "$OUT/${name}_${i}.tricks.csv" \
      > "logs/defgate_${TAG}_${name}_${i}.log" 2>&1 &
    pids+=($!)
  done
  sleep 3
  powershell -NoProfile -Command \
    "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
    > /dev/null 2>&1
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "DEFGATE_ARM $name done rc=$rc $(date)" >> "$LOG"
  return $rc
}

run_arm base hearts_ai_match.pt || { echo "DEFGATE_HALT base arm failed $(date)" >> "$LOG"; exit 1; }
run_arm cand "$CAND"            || { echo "DEFGATE_HALT cand arm failed $(date)" >> "$LOG"; exit 1; }
echo "DEFGATE_COMPLETE $(date)" >> "$LOG"
