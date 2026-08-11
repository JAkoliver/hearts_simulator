#!/bin/bash
# Phase 2 Stage E chain: the standard battery on BOTH freeze picks,
# sequential (user-signed 2026-08-11). Markers: "P2E <event>" in
# logs/p2_gates_chain.log
cd "$(dirname "$0")/.."
LOG=logs/p2_gates_chain.log
mkdir -p logs
cat /proc/$$/winpid > logs/p2_gates_chain.pid
mark() { echo "P2E $* $(date)" >> "$LOG"; }

for spec in "ep3 cand_p2_lr3e-05_ep3.pth" "ep2 cand_p2_lr3e-05_ep2.pth"; do
  set -- $spec
  TAG=$1; CAND=$2
  mark "gates $TAG start"
  if PYTHONUNBUFFERED=1 python -u run_p2_gates.py "$CAND" "$TAG" \
       >> "$LOG" 2>&1; then
    mark "gates $TAG ALL PASS"
  else
    mark "gates $TAG HALT"
  fi
done
mark COMPLETE
