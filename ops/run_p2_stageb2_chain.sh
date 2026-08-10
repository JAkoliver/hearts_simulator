#!/bin/bash
# Phase 2 Stage B-2 chain (prereg addendum, signed 2026-08-10) +
# authorized defense gate on cand_r3_probe005.pth. Sequential on
# purpose: daytime (rule 17), both stages want the GPU.
#   1. rebuild SearchEval (--compare-seed-off added)
#   2. B-2 probe run A: seed 212000200, K=256 ref, independent stream
#      (row-paired with stageb_200.csv via deterministic self-play)
#   3. B-2 probe run B: seed 212001200, same config (tightens n)
#   4. analyze_p2_stageb2.py (pairing check FIRST, registered bands)
#   5. trace cand_r3_probe005.pth -> defense gate (r1 base arm reused)
# Markers: "B2CHAIN <event>" in logs/p2_stageb2_chain.log
cd "$(dirname "$0")/.."
LOG=logs/p2_stageb2_chain.log
mkdir -p logs expert_data/p2
cat /proc/$$/winpid > logs/p2_stageb2_chain.pid
mark() { echo "B2CHAIN $* $(date)" >> "$LOG"; }

mark rebuild
cmake --build build --config Release --target SearchEval >> "$LOG" 2>&1 \
  || { mark "BUILD FAILED"; exit 1; }

probe() {  # tag seed
  local tag=$1 sd=$2
  mark "probe $tag start"
  build/Release/SearchEval --tree-selfplay \
    --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --iterations 200 --k 24 --matches 1 --seed "$sd" --cuda \
    --flat-compare "expert_data/p2/stageb2_$tag.csv" \
    --compare-k 256 --compare-seed-off 6000 \
    --record-out "expert_data/p2/stageb2_$tag.hvt" \
    > "logs/p2_stageb2_$tag.log" 2>&1 &
  local pid=$!
  sleep 3
  powershell -NoProfile -Command \
    "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" \
    > /dev/null 2>&1
  wait "$pid" || { mark "probe $tag FAILED"; exit 1; }
  PYTHONUNBUFFERED=1 python -u validate_p2_records.py \
    "expert_data/p2/stageb2_$tag.hvt" --iters 200 >> "$LOG" 2>&1
}

probe 200 212000200
probe 200b 212001200

mark "analysis start"
PYTHONUNBUFFERED=1 python -u analyze_p2_stageb2.py >> "$LOG" 2>&1
mark "analysis rc=$? (2=pairing-halt, else bands in verdict)"

mark "defense gate: trace export"
PYTHONUNBUFFERED=1 python -u export_b2_trace.py \
  cand_r3_probe005.pth cand_r3_probe005.trace.pt >> "$LOG" 2>&1 \
  || { mark "TRACE EXPORT FAILED - defense gate skipped"; exit 1; }

DIR=equity_data/exploiter_r2/gate_probe005
mkdir -p "$DIR"
cp equity_data/exploiter_r1/gate_r1t1/base_0.csv "$DIR/base_0.csv"
cp equity_data/exploiter_r1/gate_r1t1/base_1.csv "$DIR/base_1.csv"
mark "defense gate start"
bash ops/run_r2_defense_gate.sh probe005 cand_r3_probe005.trace.pt \
  || { mark "DEFENSE GATE SHARD-FAIL"; exit 1; }
PYTHONUNBUFFERED=1 python -u analyze_defense_gate.py --tag probe005 \
  --dir "$DIR" \
  --verdict equity_data/verdicts/r3_probe005_defense_gate.json \
  >> "$LOG" 2>&1
mark "defense gate analyzed rc=$?"
mark COMPLETE
