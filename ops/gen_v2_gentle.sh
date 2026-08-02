#!/bin/bash
# v2 generation, GENTLE profile, runs until the 50k-confident watcher stops
# it (kills by PID file - never by pattern, per the 07-31 self-match lesson).
# Interleaves seeded and natural chunks so both the accelerator families and
# the >=15k natural composition floor accumulate together.
cd /e/hearts_simulator || exit 1
# Windows pid, not MSYS $$ (kill-by-PID-file must target the Windows pid).
cat /proc/$$/winpid > /e/hearts_simulator/logs/gen_v2_driver.pid 2>/dev/null \
  || echo $$ > /e/hearts_simulator/logs/gen_v2_driver.pid
export HEARTS_HEADROOM=0.45
export HEARTS_SRV_MAX_ROWS=2048
GEN=./build/Release/SelfPlayGen.exe
COMMON="--match --model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
 --k 64 --k-endgame 256 --pass-k 24 --pass-candidates 12 --threads 5 --cuda --bf16"

run_chunk() {
  local name=$1 quota=$2 seed=$3; shift 3
  echo "=== CHUNK $name: $quota matches, seed $seed $* ($(date +%H:%M)) ==="
  local t0=$(date +%s)
  ( sleep 15
    powershell -NoProfile -Command "try { Get-Process SelfPlayGen -ErrorAction Stop | ForEach-Object { \$_.PriorityClass = 'BelowNormal' } } catch {}" >/dev/null 2>&1
  ) &
  $GEN $COMMON --deals "$quota" --seed "$seed" "$@" \
      --out "expert_data/v2_${name}.bin" > "logs/v2_${name}.log" 2>&1
  local rc=$?
  echo "CHUNK_${name}_RC=$rc SECONDS=$(( $(date +%s) - t0 ))"
  if [ "$rc" -ne 0 ]; then echo "CHUNK_${name}_FAILED (see logs/v2_${name}.log)"; exit 1; fi
  sleep 5
}

i=${1:-0}   # start index (resume support: avoids clobbering earlier chunks)
while true; do
  s=$((26080100 + i * 7919))
  case $((i % 6)) in
    0) run_chunk "knife_$i"  60 "$s" --start-totals 90,88,86,84 ;;
    1) run_chunk "nat_$i"    40 "$s" ;;
    2) run_chunk "leader_$i" 60 "$s" --start-totals 92,70,68,66 ;;
    3) run_chunk "nat_$i"    40 "$s" ;;
    4) run_chunk "trail_$i"  60 "$s" --start-totals 60,88,86,84 ;;
    5) run_chunk "nat_$i"    40 "$s" ;;
  esac
  i=$((i + 1))
done
