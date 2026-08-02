#!/bin/bash
# v2 generation, FULL SPEED, SIX seeded families interleaved with natural
# (widened score-state manifold coverage, adopted 2026-08-01). Runs until
# the 50k watcher stops it by PID file.
cd /e/hearts_simulator || exit 1
# Windows pid, not MSYS $$ (kill-by-PID-file must target the Windows pid).
cat /proc/$$/winpid > /e/hearts_simulator/logs/gen_v2_driver.pid 2>/dev/null \
  || echo $$ > /e/hearts_simulator/logs/gen_v2_driver.pid
unset HEARTS_HEADROOM
export HEARTS_SRV_MAX_ROWS=8192
GEN=./build/Release/SelfPlayGen.exe
COMMON="--match --model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
 --k 64 --k-endgame 256 --pass-k 24 --pass-candidates 12 --threads 14 --cuda --bf16"

run_chunk() {
  local name=$1 quota=$2 seed=$3; shift 3
  echo "=== CHUNK $name: $quota matches, seed $seed $* ($(date +%H:%M)) ==="
  local t0=$(date +%s)
  ( sleep 15
    powershell -NoProfile -Command "try { Get-Process SelfPlayGen -ErrorAction Stop | ForEach-Object { \$_.PriorityClass = 'Normal' } } catch {}" >/dev/null 2>&1
  ) &
  $GEN $COMMON --deals "$quota" --seed "$seed" "$@" \
      --out "expert_data/v2_${name}.bin" > "logs/v2_${name}.log" 2>&1
  local rc=$?
  echo "CHUNK_${name}_RC=$rc SECONDS=$(( $(date +%s) - t0 ))"
  if [ "$rc" -ne 0 ]; then echo "CHUNK_${name}_FAILED (see logs/v2_${name}.log)"; exit 1; fi
  sleep 5
}

i=${1:-45}
while true; do
  s=$((26080100 + i * 7919))
  case $((i % 12)) in
    0)  run_chunk "nat_$i"    40 "$s" ;;
    1)  run_chunk "knife_$i"  60 "$s" --start-totals 90,88,86,84 --start-jitter 2 ;;
    2)  run_chunk "nat_$i"    40 "$s" ;;
    3)  run_chunk "mid_$i"    60 "$s" --start-totals 75,73,70,68 --start-jitter 6 ;;
    4)  run_chunk "nat_$i"    40 "$s" ;;
    5)  run_chunk "leader_$i" 60 "$s" --start-totals 92,70,68,66 --start-jitter 3 ;;
    6)  run_chunk "nat_$i"    40 "$s" ;;
    7)  run_chunk "asym_$i"   60 "$s" --start-totals 85,60,55,50 --start-jitter 6 ;;
    8)  run_chunk "nat_$i"    40 "$s" ;;
    9)  run_chunk "trail_$i"  60 "$s" --start-totals 60,88,86,84 --start-jitter 3 ;;
    10) run_chunk "nat_$i"    40 "$s" ;;
    11) run_chunk "early_$i"  60 "$s" --start-totals 60,58,40,38 --start-jitter 7 ;;
  esac
  i=$((i + 1))
done
