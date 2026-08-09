#!/bin/bash
# Round-3 anchored-PPO trial (docs/exploiter_league_r3_prereg.md).
# Restore-verify discipline on BOTH sides of training; fresh Adam per
# trial; candidate archived; drift measured (measurement, not gate).
# Usage: nohup bash ops/run_r3_trial.sh <tag> > logs/r3_<tag>_nohup.log 2>&1 &
cd "$(dirname "$0")/.."
TAG=${1:?tag required}
LOG=logs/r3_trial_$TAG.log
MILESTONE=Hall_of_Fame/hearts_model_milestone_1785322724.pth
mkdir -p logs
cat /proc/$$/winpid > logs/r3_trial.pid
mark() { echo "R3TRIAL $TAG $* $(date)" >> "$LOG"; }

cp "$MILESTONE" hearts_model_final.pth
[ "$(md5sum hearts_model_final.pth | cut -c1-8)" = "8a89da90" ] \
  || { mark "HALT bad baseline restore"; exit 1; }
rm -f hearts_optimizer.pth          # fresh Adam per prereg
mark "start kl=$(python -c "import json;print(json.load(open('config.json'))['anchor_kl_coef'])")"

HEARTS_HEADROOM=${HEARTS_HEADROOM:-0.25} PYTHONUNBUFFERED=1 \
  python -u train.py >> "$LOG" 2>&1
rc=$?
mark "train done rc=$rc"
[ $rc -ne 0 ] && { mark "HALT train failed"; exit 1; }

cp hearts_model_final.pth "cand_r3_$TAG.pth"
[ -f hearts_optimizer.pth ] && cp hearts_optimizer.pth "cand_r3_$TAG.optim.pth"
mark "candidate archived cand_r3_$TAG.pth md5=$(md5sum cand_r3_$TAG.pth | cut -c1-8)"

cp "$MILESTONE" hearts_model_final.pth
[ "$(md5sum hearts_model_final.pth | cut -c1-8)" = "8a89da90" ] \
  || { mark "HALT bad baseline re-restore"; exit 1; }
mark "baseline restored"

# drift MEASUREMENT (screen bar retired; band target 0.85-0.95)
PYTHONUNBUFFERED=1 python -u drift_screen_b2.py "cand_r3_$TAG.pth" \
  --json "equity_data/verdicts/r3_drift_$TAG.json" >> "$LOG" 2>&1
mark "drift measured (see r3_drift_$TAG.json; exit code is NOT a gate)"
mark "COMPLETE"
