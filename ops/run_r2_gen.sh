#!/bin/bash
# Round-2 Phase A2 generation (docs/exploiter_league_r2_prereg.md):
# shooter CLONES attack (traced, argmax), 3 SEARCH defenders (frozen
# config, K=64 flat) - defender decisions recorded (v2 .sdrec).
#
# LOSSLESSLY PAUSABLE (prereg amendment): every shard runs --resume, so
#   PAUSE  = bash ops/pause_r2_gen.sh   (kills via PID files, never names)
#   RESUME = rerun this script - each shard trims its at-most-one partial
#            match and continues. Survives crashes and reboots too.
#
# Seeds: agg 160M+, sel 170M+, stride 1M/shard, matches +mi*1000
# (audited vs 20-51M, 70-92M, 100-115M, 130/139/141/142M, 150/155M
# smokes, 520M/620M, 720-722M).
# Usage: nohup bash ops/run_r2_gen.sh [SHARDS] [MATCHES_PER_SHARD] &
cd "$(dirname "$0")/.."
SHARDS=${1:-3}
PER=${2:-60}
EXE=build/Release/SearchEval
OUT=expert_data/defender_v1
LOG=logs/r2_gen.log
mkdir -p "$OUT" logs logs/r2_pids
cat /proc/$$/winpid > logs/r2_gen.pid
{
  echo "R2GEN_START $(date) shards=$SHARDS per=$PER"
  echo "R2GEN_MODELS probe_search=$(md5sum hearts_ai_search_match.pt | cut -c1-8) base=$(md5sum hearts_ai_match.pt | cut -c1-8) agg=$(md5sum shooter_agg_v1b.pt | cut -c1-8) sel=$(md5sum shooter_sel_v1.pt | cut -c1-8)"
} >> "$LOG"

run_mode() {  # name attacker_trace seedbase
  local name=$1 atk=$2 sbase=$3
  echo "R2GEN_MODE $name start $(date)" >> "$LOG"
  local pids=() i
  for i in $(seq 0 $((SHARDS - 1))); do
    "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
      --opponent-model hearts_ai_match.pt --shooter sel --pass-search \
      --search-defenders --attacker-model "$atk" --resume \
      --k 64 --matches "$PER" --seed $((sbase + i * 1000000)) --cuda \
      --out "$OUT/r2_${name}_${i}.csv" \
      --tricks-out "$OUT/r2_${name}_${i}.tricks.csv" \
      --record-out "$OUT/r2_${name}_${i}.sdrec" \
      > "logs/r2_${name}_${i}.log" 2>&1 &
    pids+=($!)
    cat /proc/${pids[-1]}/winpid > "logs/r2_pids/${name}_${i}.pid" 2>/dev/null
  done
  sleep 3
  powershell -NoProfile -Command \
    "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
    > /dev/null 2>&1
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  rm -f logs/r2_pids/${name}_*.pid
  echo "R2GEN_MODE $name done rc=$rc $(date)" >> "$LOG"
  return $rc
}

run_mode agg shooter_agg_v1b.pt 160000000 || { echo "R2GEN_HALT agg $(date)" >> "$LOG"; exit 1; }
run_mode sel shooter_sel_v1.pt 170000000 || { echo "R2GEN_HALT sel $(date)" >> "$LOG"; exit 1; }
echo "R2GEN_COMPLETE $(date)" >> "$LOG"
