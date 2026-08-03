#!/bin/bash
# Auto-pipeline, part 2: run after the human picks the anchor coefficient
# from the freeze report. Runs the comparative stage (16 trainings +
# 32 evals, ~23h, idempotent resume) and the registered analysis, with
# post-analysis integrity checks. Ends at AUTO_V2_COMPLETE - the
# confirmation replicate (seed 303) and the battery are MANUAL.
cd /e/hearts_simulator || exit 1
COEF="$1"
LOG=logs/auto_v2_pipeline.log
case "$COEF" in 0.25|1.0) ;; *) echo "usage: auto_v2_continue.sh 0.25|1.0"; exit 1;; esac
exec >> "$LOG" 2>&1
echo "AUTO_V2_CONTINUE coef=$COEF $(date '+%F %H:%M')"
echo "COEF=$COEF" >> logs/auto_v2_state

export ANCHOR_COEF="$COEF"
bash ops/run_v2_mix_experiment.sh --run
rc=$?
[ $rc -ne 0 ] && { echo "AUTO_V2_HALT_STEP3: driver rc=$rc (idempotent - rerun this script to resume)"; exit 1; }

# ---- step 4: analysis integrity ---------------------------------------
grep -q "CRN baseline-arm identity: OK" logs/v2mix_analysis.log \
  || { echo "AUTO_V2_HALT_STEP4: CRN baseline-arm identity NOT OK - do not trust the table"; exit 1; }
if grep -q "WARNING: eval sizes differ" logs/v2mix_analysis.log; then
  echo "AUTO_V2_HALT_STEP4: eval size mismatch (truncation?)"; exit 1
fi
nv=$(ls equity_data/verdicts/expert_iter_v2_*.json 2>/dev/null | grep -cv freeze)
[ "$nv" -ge 8 ] || { echo "AUTO_V2_HALT_STEP4: only $nv/8 verdict JSONs"; exit 1; }
[ -f docs/expert_iter_v2_results.md ] || { echo "AUTO_V2_HALT_STEP4: results doc missing"; exit 1; }
echo "AUTO_V2_STEP4_OK (CRN identity OK, 8 verdicts, results doc written)"
echo "AUTO_V2_COMPLETE $(date '+%F %H:%M') - comparative stage done. NEXT (manual): review docs/expert_iter_v2_results.md; confirmation replicate seed 303; one-shot battery."
