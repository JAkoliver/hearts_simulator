#!/bin/bash
# Round-2 GATE BATTERY (user-authorized 2026-08-09): for each candidate
# in order - defense gate (r1 base arm reused), and on PASS the gates
# 2+3 driver (match NI n=6400 + search guard n=4800). Halt-default at
# every step; a candidate that fails stops there, the next proceeds.
# Markers: R2BATTERY <event> in logs/r2_battery.log.
cd "$(dirname "$0")/.."
LOG=logs/r2_battery.log
mkdir -p logs
cat /proc/$$/winpid > logs/r2_battery.pid
mark() { echo "R2BATTERY $* $(date)" >> "$LOG"; }

defense() {  # tag trace
  local tag=$1 trace=$2
  local dir=equity_data/exploiter_r2/gate_$tag
  mkdir -p "$dir"
  cp equity_data/exploiter_r1/gate_r1t1/base_0.csv "$dir/base_0.csv"
  cp equity_data/exploiter_r1/gate_r1t1/base_1.csv "$dir/base_1.csv"
  mark "defense $tag start"
  bash ops/run_r2_defense_gate.sh "$tag" "$trace" \
    || { mark "defense $tag SHARD-FAIL"; return 2; }
  PYTHONUNBUFFERED=1 python -u analyze_defense_gate.py --tag "$tag" \
    --dir "$dir" \
    --verdict "equity_data/verdicts/r2_defense_gate_$tag.json" \
    >> "$LOG" 2>&1
}

for spec in "r2kl8ep1 cand_b2f_kl8_ep1.trace.pt cand_b2f_kl8.pth.ep1.pth" \
            "r2kl4ep1 cand_b2f_kl4_ep1.trace.pt cand_b2f_kl4.pth.ep1.pth"; do
  set -- $spec
  TAG=$1; TRACE=$2; CKPT=$3
  if defense "$TAG" "$TRACE"; then
    mark "defense $TAG PASS -> gates23"
    if PYTHONUNBUFFERED=1 python -u run_r2_gates.py "$CKPT" "$TAG" \
         >> "$LOG" 2>&1; then
      mark "gates23 $TAG ALL PASS"
    else
      mark "gates23 $TAG HALT"
    fi
  else
    mark "defense $TAG HALT"
  fi
done
mark COMPLETE
