#!/bin/bash
# Auto-pipeline, part 1 (trigger -> freeze pause). Polls the reserve
# watcher's log for V2_TARGET_REACHED, then: verifies generation is truly
# stopped, independently recounts the reserves, builds + verifies the mix
# banks, runs the recipe-freeze trainings, and PAUSES for the human
# anchor-coefficient choice (docs/expert_iter_v2_freeze_report.md).
# Every failed check emits AUTO_V2_HALT_* and stops - halt-is-default.
# Part 2 (after the human picks): ops/auto_v2_continue.sh <coef>.
cd /e/hearts_simulator || exit 1
LOG=logs/auto_v2_pipeline.log
STATE=logs/auto_v2_state
exec >> "$LOG" 2>&1
echo "AUTO_V2_ARMED $(date '+%F %H:%M') (pid $(cat /proc/$$/winpid 2>/dev/null || echo $$))"

until grep -q "V2_TARGET_REACHED" logs/watch_v2.log 2>/dev/null; do sleep 300; done
grep -q "TRIGGERED" "$STATE" 2>/dev/null && { echo "AUTO_V2_ALREADY_TRIGGERED - exiting"; exit 0; }
echo "TRIGGERED $(date '+%F %H:%M')" >> "$STATE"
echo "AUTO_V2_TRIGGERED $(date '+%F %H:%M')"
sleep 30   # let the watcher's kills settle

# ---- step 0: generation stopped + independent recount ------------------
if powershell -NoProfile -Command "Get-Process SelfPlayGen -ErrorAction Stop" >/dev/null 2>&1; then
  echo "AUTO_V2_HALT_STEP0: SelfPlayGen still running after target-reached"; exit 1
fi
python - << 'PYEOF' || { echo "AUTO_V2_HALT_STEP0: recount/boundary check failed"; exit 1; }
import glob, os, sys
import numpy as np
sys.path.insert(0, '.')
from distill import MATCH_RECORD_V2
TARGETS = {'nat': 50000, 'knife': 8400, 'mid': 8400, 'leader': 8400,
           'asym': 8400, 'trail': 8400, 'early': 8400}
conf = {k: 0 for k in TARGETS}
for f in glob.glob('expert_data/v2_*.bin'):
    sz = os.path.getsize(f)
    if sz < 32 + 848:
        print(f"deleting kill stub {f} ({sz} B)"); os.remove(f); continue
    good = 32 + ((sz - 32) // 848) * 848
    if sz != good:
        with open(f, 'r+b') as fh: fh.truncate(good)
        print(f"trimmed partial tail: {f}")
    fam = next((k for k in TARGETS if f'_{k}_' in os.path.basename(f)), None)
    if fam is None: continue
    with open(f, 'rb') as fh:
        if fh.read(4) != b'HMR2': continue
    n = (os.path.getsize(f) - 32) // 848
    if n <= 0: continue
    a = np.fromfile(f, dtype=MATCH_RECORD_V2, offset=32, count=n)
    c = ((a['flags'] & 1) == 0) & (a['second_action'] != 0xFFFF) & \
        ((a['eq_best'] - a['eq_second']) > 2.0 * a['gap_se'])
    conf[fam] += int(c.sum())
short = {k: (v, TARGETS[k]) for k, v in conf.items() if v < TARGETS[k]}
print("independent recount:", ' '.join(f"{k}={v}" for k, v in conf.items()))
if short:
    print("SHORTFALL:", short); sys.exit(1)
PYEOF
echo "AUTO_V2_STEP0_OK (generation stopped, reserves independently verified)"

# ---- step 1: build + verify mix banks ----------------------------------
PYTHONUNBUFFERED=1 python -u build_v2_mixes.py > logs/auto_v2_bankbuild.log 2>&1 \
  || { echo "AUTO_V2_HALT_STEP1: bank build failed (logs/auto_v2_bankbuild.log)"; exit 1; }
PYTHONUNBUFFERED=1 python -u verify_v2_banks.py >> logs/auto_v2_bankbuild.log 2>&1 \
  || { echo "AUTO_V2_HALT_STEP1: bank verification failed (logs/auto_v2_bankbuild.log)"; exit 1; }
echo "AUTO_V2_STEP1_OK (8 banks built, hashed and recounted)"

# ---- step 2: recipe-freeze trainings + report --------------------------
PYTHONUNBUFFERED=1 python -u freeze_v2_recipe.py > logs/auto_v2_freeze.log 2>&1
rc=$?
if [ $rc -eq 2 ]; then
  echo "AUTO_V2_HALT_STEP2: BOTH coefficients violate the entropy constraint - recipe broken, human needed"
  exit 1
elif [ $rc -ne 0 ]; then
  echo "AUTO_V2_HALT_STEP2: freeze script failed rc=$rc (logs/auto_v2_freeze.log)"
  exit 1
fi
echo "AUTO_V2_STEP2_OK (freeze report written)"
echo "AUTO_V2_AWAITING_COEF: read docs/expert_iter_v2_freeze_report.md, then run: nohup bash ops/auto_v2_continue.sh <0.25|1.0> &"
