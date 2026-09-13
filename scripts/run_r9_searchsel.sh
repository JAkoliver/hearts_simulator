#!/bin/bash
# League r9 resumable search-SEL driver (docs/exploiter_league_r9_prereg.md §3.1/§3.4).
# UNIT = ONE MATCH PER PROCESS. SearchEval is deterministic per process (verified
# 2026-09-13: two independent runs bit-identical on matches 0-4), but its search
# RNG is ONE STREAM PER PROCESS, so an in-process --resume changes the samples of
# the matches after the resume point. Running every match as its own process
# makes a killed match simply replay from scratch, bit-identical to an
# uninterrupted run BY CONSTRUCTION - no engine change, frozen binary untouched.
# A match is COMPLETE when its .done marker exists (written only on rc=0);
# partial outputs of a killed match are deleted and replayed. Deals of match mi
# of shard s are exactly the r7/r8 convention (env seed = shard seed + mi*1000),
# so rows pair with the r7 base rows on identical deals; the search-sample
# stream per match starts fresh (statistically equivalent, not bitwise, to the
# old 32-matches-per-process convention - recorded in the prereg freeze notes).
# Modes:
#   transfer <tag> <trace> : shards 12..15 x 32 matches (seeds 732260806+(s-12)*1e6+mi*1000)
#   base                   : same matches, the PROMOTED ensemble's fused trace (Stage 0, once)
#   bank <tag> <trace> <N> : N matches vs <trace>, seeds 140,000,000 + m*10,000, --record-out
# <= R9_CONC (default 2) concurrent SearchEval; BelowNormal unless HEARTS_NO_LOWPRI=1.
# Pause = kill the tree; re-run the same command to resume.
cd "$(dirname "$0")/.."
MODE=${1:?transfer|base|bank}
EXE=build/Release/SearchEval.exe
CONC=${R9_CONC:-2}
PROMOTED_TRACE=hybrid_champ_arma_moonhead_0p1_882_v2.pt

UNITS=()   # name:seed
case "$MODE" in
  transfer) TAG=${2:?tag}; TRACE=${3:?trace}; OUT=equity_data/exploiter_r9/transfer/$TAG; RECORD=0
            for s in 12 13 14 15; do for ((mi = 0; mi < 32; mi++)); do UNITS+=("s${s}m$(printf %02d $mi):$((732260806 + (s - 12) * 1000000 + mi * 1000))"); done; done ;;
  base)     TAG=promoted; TRACE=$PROMOTED_TRACE; OUT=equity_data/exploiter_r9/transfer/promoted_base; RECORD=0
            for s in 12 13 14 15; do for ((mi = 0; mi < 32; mi++)); do UNITS+=("s${s}m$(printf %02d $mi):$((732260806 + (s - 12) * 1000000 + mi * 1000))"); done; done ;;
  bank)     TAG=${2:?tag}; TRACE=${3:?trace}; N=${4:?matches}; OUT=equity_data/exploiter_r9/bank/$TAG; RECORD=1
            for ((m = 0; m < N; m++)); do UNITS+=("m$(printf %03d $m):$((140000000 + m * 10000))"); done ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
mkdir -p "$OUT"; LOG=$OUT/driver.log
echo "R9_START $(date) mode=$MODE tag=$TAG trace=$(md5sum "$TRACE" | cut -c1-8) exe=$(md5sum $EXE | cut -c1-8) search=$(md5sum hearts_ai_search_match.pt | cut -c1-8) equity=$(md5sum hearts_equity.pt | cut -c1-8) units=${#UNITS[@]} conc=$CONC" >> "$LOG"

lowpri() {
  [ "${HEARTS_NO_LOWPRI:-0}" = "1" ] && return
  powershell -NoProfile -Command "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" > /dev/null 2>&1
}
run_unit() {  # name seed
  local name=$1 seed=$2 rc
  rm -f "$OUT/$name.csv" "$OUT/$name.tricks.csv" "$OUT/phaseB_sel_$name.sdrec" "$OUT/$name.done"
  # bank records are named phaseB_sel_<unit>.sdrec so train_shooter.py --data <dir> finds them (its v1 glob)
  local extra=(); [ "$RECORD" = "1" ] && extra=(--record-out "$OUT/phaseB_sel_$name.sdrec")
  "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --opponent-model "$TRACE" --shooter sel --pass-search --k 64 --matches 1 \
    --seed "$seed" --cuda --out "$OUT/$name.csv" --tricks-out "$OUT/$name.tricks.csv" \
    "${extra[@]}" > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$OUT/$name.csv" ]; then
    echo "$seed $(date +%s)" > "$OUT/$name.done"
  fi
  echo "unit $name DONE rc=$rc $(date +%T)" >> "$LOG"
  return $rc
}

fail=0; started=0
for entry in "${UNITS[@]}"; do
  name=${entry%%:*}; seed=${entry##*:}
  [ -f "$OUT/$name.done" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n || fail=1; done
  run_unit "$name" "$seed" &
  started=$((started + 1))
  sleep 2; lowpri
done
wait || fail=1
missing=0
for entry in "${UNITS[@]}"; do name=${entry%%:*}; [ -f "$OUT/$name.done" ] || missing=$((missing + 1)); done
echo "R9_END mode=$MODE tag=$TAG started=$started missing=$missing $(date)" >> "$LOG"
if [ "$missing" = "0" ]; then echo "R9_DONE mode=$MODE tag=$TAG $(date)" >> "$LOG"; exit 0; fi
echo "R9_INCOMPLETE mode=$MODE tag=$TAG missing=$missing (re-run to resume) $(date)" >> "$LOG"; exit 1
