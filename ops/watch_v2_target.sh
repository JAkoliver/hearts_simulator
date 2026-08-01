#!/bin/bash
# Mix-experiment reserve watcher (supersedes the 50k-total stop, 2026-08-01):
# stop when natural-confident >= 50,000 AND every seeded family >= 8,400
# confident play-phase records (covers all five 50k mixes incl. natural-only
# and balanced seeded-spread). Kills by PID file + exe name only.
cd /e/hearts_simulator || exit 1
while true; do
  line=$(python - << 'PYEOF'
import glob, os
import numpy as np
DT = np.dtype([('obs','u1',556),('mask','u1',52),('labels','u1',156),('pi','u1',52),
               ('action','<u2'),('seat','<u2'),('reward','<f4'),
               ('eq_best','<f4'),('eq_second','<f4'),('gap_se','<f4'),
               ('second_action','<u2'),('n_dets','<u2'),('match_id','<u4'),
               ('flags','<u2'),('reserved','<u2')])
FAMS = ['nat', 'knife', 'mid', 'leader', 'asym', 'trail', 'early']
conf = {k: 0 for k in FAMS}
tot = 0
for f in glob.glob('expert_data/v2_*.bin'):
    b = os.path.basename(f)
    fam = next((k for k in FAMS if f'_{k}_' in b), None)
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
    conf[fam] += int(c.sum()); tot += len(a)
seeded = {k: v for k, v in conf.items() if k != 'nat'}
parts = ' '.join(f'{k}={v}' for k, v in conf.items())
print(f"V2PROGRESS total={tot} {parts}")
ok = conf['nat'] >= 50000 and all(v >= 8400 for v in seeded.values())
print("V2DONE" if ok else "V2NOTYET")
PYEOF
)
  echo "$line" | head -1
  if echo "$line" | grep -q "^V2DONE"; then
    if [ -f logs/gen_v2_driver.pid ]; then
      powershell -NoProfile -Command "Stop-Process -Id $(cat logs/gen_v2_driver.pid) -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
    fi
    powershell -NoProfile -Command "Stop-Process -Name SelfPlayGen -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
    echo "V2_TARGET_REACHED $(date +%H:%M)"
    exit 0
  fi
  sleep 900
done
