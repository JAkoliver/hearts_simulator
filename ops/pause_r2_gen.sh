#!/bin/bash
# LOSSLESS PAUSE for round-2 generation: kill by PID FILE (never by
# name), driver first so it can't launch the next mode, then the shards.
# Resume = rerun ops/run_r2_gen.sh (shards trim their one partial match
# and continue).
cd "$(dirname "$0")/.."
if [ -f logs/r2_gen.pid ]; then
  taskkill //F //PID "$(cat logs/r2_gen.pid)" 2>/dev/null \
    && echo "driver stopped (pid $(cat logs/r2_gen.pid))"
  rm -f logs/r2_gen.pid
else
  echo "no driver pid file - driver not running?"
fi
for f in logs/r2_pids/*.pid; do
  [ -e "$f" ] || continue
  taskkill //F //PID "$(cat "$f")" 2>/dev/null \
    && echo "shard $(basename "$f" .pid) stopped (pid $(cat "$f"))"
  rm -f "$f"
done
echo "R2GEN_PAUSED $(date)" >> logs/r2_gen.log
echo "paused. resume with: nohup bash ops/run_r2_gen.sh > logs/r2_nohup.log 2>&1 &"
