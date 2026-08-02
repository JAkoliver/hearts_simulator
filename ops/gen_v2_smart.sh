#!/bin/bash
# v2 generation, DEMAND-AWARE (2026-08-01): before each chunk, count per-family
# confident reserves and generate ONLY families still below target
# (nat 50,000; each seeded family 8,400). Filled families drop out of the
# rotation; when only natural remains, every chunk is natural. Full speed;
# profile switches still via kill+relaunch. Jittered seeded starts.
cd /e/hearts_simulator || exit 1
# Record the WINDOWS pid (taskkill/Stop-Process target), not the MSYS $$ —
# they differ; a stale MSYS pid makes kill-by-PID-file miss (found 2026-08-01).
cat /proc/$$/winpid > /e/hearts_simulator/logs/gen_v2_driver.pid 2>/dev/null \
  || echo $$ > /e/hearts_simulator/logs/gen_v2_driver.pid
unset HEARTS_HEADROOM
export HEARTS_SRV_MAX_ROWS=8192
GEN=./build/Release/SelfPlayGen.exe
COMMON="--match --model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
 --k 64 --k-endgame 256 --pass-k 24 --pass-candidates 12 --threads 14 --cuda --bf16"

# Echoes space-separated families still below target, natural first.
needs() {
python - << 'PYEOF'
import glob, os
import numpy as np
DT = np.dtype([('obs','u1',556),('mask','u1',52),('labels','u1',156),('pi','u1',52),
               ('action','<u2'),('seat','<u2'),('reward','<f4'),
               ('eq_best','<f4'),('eq_second','<f4'),('gap_se','<f4'),
               ('second_action','<u2'),('n_dets','<u2'),('match_id','<u4'),
               ('flags','<u2'),('reserved','<u2')])
TARGETS = {'nat': 50000, 'knife': 8400, 'mid': 8400, 'leader': 8400,
           'asym': 8400, 'trail': 8400, 'early': 8400}
conf = {k: 0 for k in TARGETS}
for f in glob.glob('expert_data/v2_*.bin'):
    b = os.path.basename(f)
    fam = next((k for k in TARGETS if f'_{k}_' in b), None)
    if fam is None: continue
    try:
        with open(f, 'rb') as fh:
            if fh.read(4) != b'HMR2': continue
        n = (os.path.getsize(f) - 32) // 848
        if n <= 0: continue
        a = np.fromfile(f, dtype=DT, offset=32, count=n)
    except Exception:
        continue
    c = ((a['flags'] & 1) == 0) & (a['second_action'] != 0xFFFF) & \
        ((a['eq_best'] - a['eq_second']) > 2.0 * a['gap_se'])
    conf[fam] += int(c.sum())
print(' '.join(k for k, t in TARGETS.items() if conf[k] < t))
PYEOF
}

seeded_args() {
  case $1 in
    knife)  echo "--start-totals 90,88,86,84 --start-jitter 2" ;;
    mid)    echo "--start-totals 75,73,70,68 --start-jitter 6" ;;
    leader) echo "--start-totals 92,70,68,66 --start-jitter 3" ;;
    asym)   echo "--start-totals 85,60,55,50 --start-jitter 6" ;;
    trail)  echo "--start-totals 60,88,86,84 --start-jitter 3" ;;
    early)  echo "--start-totals 60,58,40,38 --start-jitter 7" ;;
  esac
}

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

i=${1:-70}
sidx=0
while true; do
  need=$(needs)
  [ -z "$need" ] && { echo "ALL_RESERVES_MET $(date +%H:%M)"; exit 0; }
  s=$((26080100 + i * 7919))
  # Alternate: natural (if needed), then the next needed seeded family.
  if [ $((i % 2)) -eq 0 ] && echo "$need" | grep -qw nat; then
    run_chunk "nat_$i" 40 "$s"
  else
    seeded=$(echo "$need" | tr ' ' '\n' | grep -v '^nat$' | head -20)
    if [ -z "$seeded" ]; then
      run_chunk "nat_$i" 40 "$s"
    else
      nfam=$(echo "$seeded" | wc -l)
      fam=$(echo "$seeded" | sed -n "$(( (sidx % nfam) + 1 ))p")
      sidx=$((sidx + 1))
      run_chunk "${fam}_$i" 60 "$s" $(seeded_args "$fam")
    fi
  fi
  i=$((i + 1))
done
