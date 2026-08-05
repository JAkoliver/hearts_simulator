#!/bin/bash
# Exploiter league Phase A data collection (docs/exploiter_league_prereg.md).
#
# Three registered combos, run SEQUENTIALLY (bounded GPU load), shards
# parallel within each: AGG vs baseline field, SEL vs baseline field,
# SEL vs v4-m10 field. Shooter = deployed match-aware search stack
# (hearts_ai_search_match.pt + hearts_equity.pt, K=64 / K_endgame=256,
# pass search) with the moon objective = shooter_v1 when frozen.
#
# Seeds (FRESH space, collision-audited vs generation [20-51M] and eval
# blocks [520M/620M] 2026-08-05): agg_base 70M+, sel_base 80M+,
# sel_v4 90M+; shard i at base + i*1,000,000, matches at +mi*1000.
#
# Usage: nohup bash ops/run_phaseA_shooter.sh [SHARDS] [MATCHES_PER_SHARD] &
# Kill: taskkill by logs/phaseA_driver.pid (winpid), then sweep
# SearchEval.exe by NAME (never by command-line pattern).
cd "$(dirname "$0")/.."
mkdir -p equity_data/exploiter_r1 logs
SHARDS=${1:-6}
PER=${2:-67}
EXE=build/Release/SearchEval
BASE_DEF=hearts_ai_match.pt
V4_DEF=hearts_ai_grandmaster_v4m10.pt
SHOOT=hearts_ai_search_match.pt
EQ=hearts_equity.pt
LOG=logs/phaseA_shooter.log
cat /proc/$$/winpid > logs/phaseA_driver.pid
{
  echo "PHASEA_START $(date) shards=$SHARDS per=$PER"
  echo "PHASEA_MODELS shooter=$(md5sum $SHOOT | cut -c1-8) eq=$(md5sum $EQ | cut -c1-8) base=$(md5sum $BASE_DEF | cut -c1-8) v4=$(md5sum $V4_DEF | cut -c1-8)"
  echo "PHASEA_SEEDS agg_base=70000000 sel_base=80000000 sel_v4=90000000 stride=1000000"
} >> "$LOG"

run_combo() {  # name mode defender seedbase
  local name=$1 mode=$2 def=$3 sbase=$4
  echo "PHASEA_COMBO $name start $(date)" >> "$LOG"
  local pids=()
  local i
  for i in $(seq 0 $((SHARDS - 1))); do
    "$EXE" --search-model "$SHOOT" --equity-model "$EQ" \
      --opponent-model "$def" --shooter "$mode" --pass-search \
      --k 64 --k-endgame 256 --matches "$PER" --seed $((sbase + i * 1000000)) \
      --cuda \
      --out "equity_data/exploiter_r1/phaseA_${name}_${i}.csv" \
      --tricks-out "equity_data/exploiter_r1/phaseA_${name}_${i}.tricks.csv" \
      > "logs/phaseA_${name}_${i}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  echo "PHASEA_COMBO $name done rc=$rc $(date)" >> "$LOG"
  return $rc
}

run_combo agg_base agg "$BASE_DEF" 70000000 || { echo "PHASEA_HALT agg_base" >> "$LOG"; exit 1; }
run_combo sel_base sel "$BASE_DEF" 80000000 || { echo "PHASEA_HALT sel_base" >> "$LOG"; exit 1; }
run_combo sel_v4 sel "$V4_DEF" 90000000 || { echo "PHASEA_HALT sel_v4" >> "$LOG"; exit 1; }

python analyze_phasea.py --prefix equity_data/exploiter_r1/phaseA --write >> "$LOG" 2>&1
echo "PHASEA_COMPLETE $(date)" >> "$LOG"
