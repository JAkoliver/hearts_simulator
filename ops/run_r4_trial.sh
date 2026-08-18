#!/bin/bash
# Exploiter league ROUND 4 trial driver (docs/exploiter_league_r4_prereg.md §4.1).
# One factorial cell per invocation: anchored match-PPO from 8a89da90 with the
# r3 recipe (config_r4_base.json, md5-frozen) and exactly two overrides:
# anchor_kl_coef (lambda) and shooter_share (s). Restore-verify on both sides,
# fresh Adam, candidate + mid-trial snapshot archived, drift measured, and the
# fast defense probe run on BOTH snapshots (mid + end). config.json (a WIP file
# in this tree) is backed up and restored byte-for-byte.
#
# Usage: nohup bash ops/run_r4_trial.sh <cell> <lambda> <share> [block_credit_b] > logs/r4_<cell>_nohup.log 2>&1 &
#   e.g. bash ops/run_r4_trial.sh A 0.05 0.15          (round 1)
#        bash ops/run_r4_trial.sh R1 0.05 0.15 2.0     (Addendum R, block credit b)
cd "$(dirname "$0")/.."
CELL=${1:?cell tag required (A/B/C/D)}
LAMBDA=${2:?anchor_kl_coef required}
SHARE=${3:?shooter_share required}
BCREDIT=${4:-0}      # Addendum R block_credit_b (0 = round-1 recipe)
# League r5 (docs/exploiter_league_r5_prereg.md): R5_INIT=<ckpt> starts the
# trial from that checkpoint instead of the champion (the ext-adapter init,
# md5 R5_INIT_MD5), and R5_STRENGTH=1 adds the paired neutral-raw strength
# read of the END candidate vs the champion (n=5000). Unset = r4 behaviour.
R5_INIT=${R5_INIT:-}
R5_INIT_MD5=${R5_INIT_MD5:-}
R5_STRENGTH=${R5_STRENGTH:-0}
LOG=logs/r4_trial_$CELL.log
MILESTONE=Hall_of_Fame/hearts_model_milestone_1785322724.pth
BASE_CFG=config_r4_base.json
BASE_CFG_MD5=7ac03df82afd979f4e29ca60fe2e5e4a
PROBE_SEED=${PROBE_SEED:-740000000}
PROBE_N=${PROBE_N:-1000}
mkdir -p logs equity_data/exploiter_r4
cat /proc/$$/winpid > logs/r4_trial.pid
mark() { echo "R4TRIAL $CELL $* $(date)" >> "$LOG"; }

# --- frozen-instrument checks -------------------------------------------------
[ "$(md5sum $BASE_CFG | cut -c1-32)" = "$BASE_CFG_MD5" ] \
  || { mark "HALT config_r4_base.json md5 mismatch"; exit 1; }
[ "$(md5sum $MILESTONE | cut -c1-8)" = "8a89da90" ] \
  || { mark "HALT champion md5 mismatch"; exit 1; }

# --- config: back up WIP, write cell config ----------------------------------
cp config.json "equity_data/exploiter_r4/config_json_backup_$CELL.json"
python - "$LAMBDA" "$SHARE" "$BCREDIT" <<'PY'
import json, sys
c = json.load(open('config_r4_base.json'))
c['anchor_kl_coef'] = float(sys.argv[1])
c['shooter_share'] = float(sys.argv[2])
c['block_credit_b'] = float(sys.argv[3])
json.dump(c, open('config.json', 'w'), indent=1)
PY
cp config.json "equity_data/exploiter_r4/config_$CELL.json"
mark "config written lambda=$LAMBDA share=$SHARE block_credit_b=$BCREDIT md5=$(md5sum config.json | cut -c1-8)"

restore_cfg() { cp "equity_data/exploiter_r4/config_json_backup_$CELL.json" config.json; mark "config.json WIP restored"; }
trap restore_cfg EXIT

# --- baseline restore (or r5 init), fresh Adam -------------------------------
if [ -n "$R5_INIT" ]; then
  cp "$R5_INIT" hearts_model_final.pth
  [ "$(md5sum hearts_model_final.pth | cut -c1-8)" = "$R5_INIT_MD5" ] || { mark "HALT bad r5 init restore ($R5_INIT expected $R5_INIT_MD5)"; exit 1; }
  mark "r5 init: $R5_INIT md5=$R5_INIT_MD5"
else
  cp "$MILESTONE" hearts_model_final.pth
  [ "$(md5sum hearts_model_final.pth | cut -c1-8)" = "8a89da90" ] || { mark "HALT bad baseline restore"; exit 1; }
fi
rm -f hearts_optimizer.pth hearts_model_mid.pth
mark "start lambda=$LAMBDA share=$SHARE block_credit_b=$BCREDIT (fresh Adam, HEADROOM ${HEARTS_HEADROOM:-0.25})"

HEARTS_HEADROOM=${HEARTS_HEADROOM:-0.25} PYTHONUNBUFFERED=1 \
  python -u train.py >> "$LOG" 2>&1
rc=$?
mark "train done rc=$rc"
[ $rc -ne 0 ] && { mark "HALT train failed"; exit 1; }

# --- archive candidate + mid snapshot, restore baseline ------------------------
cp hearts_model_final.pth "cand_r4_$CELL.pth"
[ -f hearts_optimizer.pth ] && cp hearts_optimizer.pth "cand_r4_$CELL.optim.pth"
[ -f hearts_model_mid.pth ] && cp hearts_model_mid.pth "cand_r4_${CELL}_mid.pth"
mark "candidate archived cand_r4_$CELL.pth md5=$(md5sum cand_r4_$CELL.pth | cut -c1-8) mid=$( [ -f cand_r4_${CELL}_mid.pth ] && md5sum cand_r4_${CELL}_mid.pth | cut -c1-8 || echo none)"

cp "$MILESTONE" hearts_model_final.pth
[ "$(md5sum hearts_model_final.pth | cut -c1-8)" = "8a89da90" ] \
  || { mark "HALT bad baseline re-restore"; exit 1; }
mark "baseline restored"

# --- drift MEASUREMENT (band 5-15% informs; not a gate) -----------------------
if [ -z "$R5_INIT" ]; then
  PYTHONUNBUFFERED=1 python -u drift_screen_b2.py "cand_r4_$CELL.pth" --json "equity_data/verdicts/r4_drift_$CELL.json" >> "$LOG" 2>&1
  mark "drift measured (r4_drift_$CELL.json; exit code is NOT a gate)"
fi
# r5 shared drift instrument (556 and 882 candidates alike; v6 holdout threat-dead)
PYTHONUNBUFFERED=1 python -u drift_screen_v6holdout.py "cand_r4_$CELL.pth" --json "equity_data/verdicts/r5_driftv6_$CELL.json" >> "$LOG" 2>&1
mark "drift (v6 holdout) measured (r5_driftv6_$CELL.json)"

# --- fast defense probe: mid + end snapshots (registered mechanism reading) ---
NETS="cand_r4_$CELL.pth"
[ -f "cand_r4_${CELL}_mid.pth" ] && NETS="cand_r4_${CELL}_mid.pth $NETS"
PYTHONUNBUFFERED=1 python -u defense_probe_fast.py --nets $NETS \
  --matches "$PROBE_N" --workers 12 --seed "$PROBE_SEED" \
  --out "equity_data/exploiter_r4/fastprobe_$CELL.csv" \
  --json "equity_data/verdicts/r4_fastprobe_$CELL.json" >> "$LOG" 2>&1
mark "fast probe done (r4_fastprobe_$CELL.json)"

# --- r5: paired strength of the END candidate vs the champion (guardrail/informs)
if [ "$R5_STRENGTH" = "1" ]; then
  PYTHONUNBUFFERED=1 python -u neutral_raw_eval.py --cand "cand_r4_$CELL.pth" --base "$MILESTONE" --deals 5000 --workers 12 >> "$LOG" 2>&1
  mark "paired strength vs champion done (see log: Neutral raw delta)"
fi
mark "COMPLETE"
