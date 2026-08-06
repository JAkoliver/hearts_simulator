#!/bin/bash
# Phase B generation (docs/exploiter_league_prereg.md): search-shooter vs
# baseline matches WITH decision recording (--record-out), both settings,
# for distilling shooter_agg_v1 / shooter_sel_v1.
#
# Seeds: fresh space, collision-audited 2026-08-06 vs generation
# [20-51M], Phase A [70-92M], eval blocks [520M/620M]:
#   agg 100M+, sel 110M+, stride 1M/shard, matches at +mi*1000.
# Instrument spec unchanged from the frozen probe: K=64 flat, pass search.
#
# Usage: nohup bash ops/run_phaseB_gen.sh [SHARDS] [MATCHES_PER_SHARD] &
cd "$(dirname "$0")/.."
mkdir -p expert_data/shooter_v1 logs
SHARDS=${1:-3}
PER=${2:-170}
EXE=build/Release/SearchEval
LOG=logs/phaseB_gen.log
cat /proc/$$/winpid > logs/phaseB_driver.pid
{
  echo "PHASEB_START $(date) shards=$SHARDS per=$PER"
  echo "PHASEB_MODELS shooter=$(md5sum hearts_ai_search_match.pt | cut -c1-8) base=$(md5sum hearts_ai_match.pt | cut -c1-8)"
  echo "PHASEB_SEEDS agg=100000000 sel=110000000 stride=1000000"
} >> "$LOG"

run_mode() {  # name mode seedbase
  local name=$1 mode=$2 sbase=$3
  echo "PHASEB_MODE $name start $(date)" >> "$LOG"
  local pids=() i
  for i in $(seq 0 $((SHARDS - 1))); do
    "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
      --opponent-model hearts_ai_match.pt --shooter "$mode" --pass-search \
      --k 64 --matches "$PER" --seed $((sbase + i * 1000000)) --cuda \
      --out "expert_data/shooter_v1/phaseB_${name}_${i}.csv" \
      --tricks-out "expert_data/shooter_v1/phaseB_${name}_${i}.tricks.csv" \
      --record-out "expert_data/shooter_v1/phaseB_${name}_${i}.sdrec" \
      > "logs/phaseB_${name}_${i}.log" 2>&1 &
    pids+=($!)
  done
  sleep 3
  powershell -NoProfile -Command \
    "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
    > /dev/null 2>&1
  local rc=0 p
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  echo "PHASEB_MODE $name done rc=$rc $(date)" >> "$LOG"
  return $rc
}

run_mode agg agg 100000000 || { echo "PHASEB_HALT agg" >> "$LOG"; exit 1; }
run_mode sel sel 110000000 || { echo "PHASEB_HALT sel" >> "$LOG"; exit 1; }
echo "PHASEB_GEN_COMPLETE $(date)" >> "$LOG"
