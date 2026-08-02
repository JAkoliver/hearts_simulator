#!/bin/bash
# Expert-iteration v2 mix experiment driver (docs/expert_iter_v2_prereg.md):
# 5 mixes x 2 training replicates x 2 disjoint seed blocks n=3200, CRN.
# Default is PLAN mode (prints the schedule, runs nothing); pass --run to
# execute. Launcher discipline per rules #8: unbuffered, file-logged.
#
# PREREQUISITES (checked at --run):
#   - mix banks built: python build_v2_mixes.py  (after reserves land)
#   - ANCHOR_COEF set from the holdout recipe freeze (prereg: {0.25, 1.0})
#   - baseline md5 verified (explicit snapshot, rule #3 -- never
#     hearts_model_final.pth)
cd /e/hearts_simulator || exit 1

# ---- frozen experiment config ------------------------------------------
BASE=Hall_of_Fame/hearts_model_milestone_1785322724.pth   # 8a89da90 (5th)
BASE_MD5=8a89da90d522fe51dff4ae2fc8170961
MIXES="a_nat60 b_even50 c_seed65 d_natonly e_seedspread"
REP_SEEDS="101 202"          # confirmation replicate reserves seed 303
# Block seeds: disjoint from each other (100M apart >> the ~14.2M span a
# block's 12 workers cover) AND far above the generation seed space
# (~26-30M, gen_v2 launchers) so no eval deal sequence can coincide with
# a training-bank deal. The confirmation BATTERY must use fresh seeds
# outside both blocks (prereg: fresh-seed battery).
BLOCK_SEEDS="520260810 620260810"
N_MATCHES=3200
WORKERS=12                   # PINNED: pairing depends on worker count
EPOCHS=3
HOLDOUT=0.10                 # by-match (mix banks are match-contiguous)
ANCHOR_COEF="${ANCHOR_COEF:-UNSET}"   # export ANCHOR_COEF=... after recipe freeze
MIN_CONF=2.0                 # frozen prereg sigma
EVAL_DIR=equity_data/expert_iter_v2
MIX_DIR=expert_data/mixes
# ------------------------------------------------------------------------

echo "=== v2 mix experiment ($(date '+%F %H:%M')) ==="
echo "baseline: $BASE (expect md5 $BASE_MD5)"
echo "plan: ${MIXES// /,} x reps(${REP_SEEDS// /,}) -> train;"
echo "      each candidate x blocks(${BLOCK_SEEDS// /,}) n=$N_MATCHES w=$WORKERS -> eval"
echo "      (= 10 trainings, 20 evals ~40min each, + null calibration)"

if [ "$1" != "--run" ]; then
  echo "PLAN MODE ONLY. Re-run with --run to execute."
  for mix in $MIXES; do
    for rs in $REP_SEEDS; do
      echo "  train: distill --data $MIX_DIR/$mix.bin --train-seed $rs -> cand_v2_${mix}_r${rs}.pth"
      for bi in $BLOCK_SEEDS; do
        echo "    eval: match_eval n=$N_MATCHES seed=$bi -> $EVAL_DIR/eval_${mix}_r${rs}_b${bi}.csv"
      done
    done
  done
  echo "  then: python analyze_v2_mixes.py --eval-dir $EVAL_DIR --write-verdicts --write-results-doc"
  exit 0
fi

# ---- preflight ----------------------------------------------------------
[ "$ANCHOR_COEF" = "UNSET" ] && { echo "ABORT_CONFIG: export ANCHOR_COEF first (recipe freeze)"; exit 1; }
md5=$(md5sum "$BASE" | cut -d' ' -f1)
[ "$md5" = "$BASE_MD5" ] || { echo "ABORT_CONFIG: baseline md5 $md5 != $BASE_MD5"; exit 1; }
for mix in $MIXES; do
  [ -f "$MIX_DIR/$mix.bin" ] || { echo "ABORT_CONFIG: missing $MIX_DIR/$mix.bin (run build_v2_mixes.py)"; exit 1; }
done
# Never share the machine with generation (rule #14 spirit: both arms of
# every eval share ALL hardware conditions; a generator half-loading the
# box mid-experiment would also wreck durations).
if powershell -NoProfile -Command "Get-Process SelfPlayGen -ErrorAction Stop" >/dev/null 2>&1; then
  echo "ABORT_CONFIG: SelfPlayGen is running -- generation must be stopped first"
  exit 1
fi
mkdir -p "$EVAL_DIR" logs

# Null calibration (project convention): an arm against itself must give
# exact-zero paired deltas.
echo "--- null calibration (base vs base, n=24) ---"
PYTHONUNBUFFERED=1 python -u match_eval.py --cand "$BASE" --base "$BASE" \
  --matches 24 --workers 4 --seed 999 \
  --csv-out "$EVAL_DIR/null_calibration.csv" > logs/v2mix_null.log 2>&1
python - << 'PYEOF' || { echo "NULL_CALIBRATION_FAILED"; exit 1; }
rows = [l.split(',') for l in open('equity_data/expert_iter_v2/null_calibration.csv')
        if not l.startswith(('#', 'idx'))]
bad = [r for r in rows if r[3] != r[8]]
raise SystemExit(1 if bad else print(f"null calibration OK ({len(rows)} matches, all deltas zero)"))
PYEOF

# ---- train + evaluate ---------------------------------------------------
for mix in $MIXES; do
  for rs in $REP_SEEDS; do
    cand="cand_v2_${mix}_r${rs}.pth"
    if [ ! -f "$cand" ]; then
      echo "--- train $cand ($(date '+%H:%M')) ---"
      PYTHONUNBUFFERED=1 python -u distill.py \
        --data "$MIX_DIR/$mix.bin" --match --arch v5 \
        --init "$BASE" --out "$cand" \
        --epochs "$EPOCHS" --holdout "$HOLDOUT" \
        --min-confidence "$MIN_CONF" \
        --anchor-coef "$ANCHOR_COEF" --anchor-model "$BASE" \
        --train-seed "$rs" > "logs/v2mix_train_${mix}_r${rs}.log" 2>&1
      rc=$?
      [ $rc -ne 0 ] && { echo "TRAIN_FAILED $cand rc=$rc"; exit 1; }
    fi
    for bi in $BLOCK_SEEDS; do
      out="$EVAL_DIR/eval_${mix}_r${rs}_b${bi}.csv"
      # Resume integrity: a crash-truncated CSV must re-run, not be
      # silently accepted (the analyzer intersects units across evals,
      # so a short file would shrink EVERY mix's n without erroring).
      if [ -f "$out" ]; then
        lines=$(wc -l < "$out")
        if [ "$lines" -ne $((N_MATCHES + 2)) ]; then
          echo "INCOMPLETE $out ($lines lines, want $((N_MATCHES + 2))) -- re-running"
          rm -f "$out"
        fi
      fi
      if [ ! -f "$out" ]; then
        echo "--- eval $mix r$rs block $bi ($(date '+%H:%M')) ---"
        PYTHONUNBUFFERED=1 python -u match_eval.py \
          --cand "$cand" --base "$BASE" \
          --matches "$N_MATCHES" --workers "$WORKERS" --seed "$bi" \
          --csv-out "$out" > "logs/v2mix_eval_${mix}_r${rs}_b${bi}.log" 2>&1
        rc=$?
        [ $rc -ne 0 ] && { echo "EVAL_FAILED $out rc=$rc"; exit 1; }
        echo "EVAL_DONE $out"
      fi
    done
  done
done

echo "--- analysis ($(date '+%H:%M')) ---"
PYTHONUNBUFFERED=1 python -u analyze_v2_mixes.py --eval-dir "$EVAL_DIR" \
  --write-verdicts --write-results-doc > logs/v2mix_analysis.log 2>&1
cat logs/v2mix_analysis.log
echo "MIX_EXPERIMENT_COMPLETE $(date '+%F %H:%M')"
