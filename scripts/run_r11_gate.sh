#!/bin/bash
# League r11 SEL defense gate driver (docs/exploiter_league_r11_prereg.md §5 item 2):
# the r9 one-match-per-process convention (scripts/run_r9_searchsel.sh) on the
# r7 gate shards. Deals of match mi of shard s: env seed = 720260806 + s*1e6 +
# mi*1000 (the r7 convention), so rows pair with the PROMOTED ensemble's r7 rows
# (equity_data/exploiter_r4/r7_gate/cand_<s>.csv) on identical deals.
# Modes:
#   ref  : ONE match (shard 0, mi 0) with the PROMOTED trace -> gate_ref/ ; the
#          gate-fires check compares it (and E25's s0m00) with the r7 rows
#   gate : E25 trace, shards 0,1,4..11 x 32 matches = 320 units -> gate/
# <= R11_CONC (default 2) concurrent SearchEval; BelowNormal unless HEARTS_NO_LOWPRI=1.
# Pause = kill the tree; re-run the same command to resume (.done markers).
cd "$(dirname "$0")/.."
MODE=${1:?ref|gate}
EXE=build/Release/SearchEval.exe
CONC=${R11_CONC:-2}
PROMOTED_TRACE=hybrid_champ_arma_moonhead_0p1_882_v2.pt
E25_TRACE=equity_data/exploiter_r11/r11_E25_ensemble_882.pt
UNITS=()
case "$MODE" in
  ref)  TRACE=$PROMOTED_TRACE; OUT=equity_data/exploiter_r11/gate_ref; UNITS+=("s0m00:720260806") ;;
  gate) TRACE=$E25_TRACE; OUT=equity_data/exploiter_r11/gate
        for s in 0 1 4 5 6 7 8 9 10 11; do for ((mi = 0; mi < 32; mi++)); do UNITS+=("s${s}m$(printf %02d $mi):$((720260806 + s * 1000000 + mi * 1000))"); done; done ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
mkdir -p "$OUT"; LOG=$OUT/driver.log
echo "R11_START $(date) mode=$MODE trace=$(md5sum "$TRACE" | cut -c1-8) exe=$(md5sum $EXE | cut -c1-8) search=$(md5sum hearts_ai_search_match.pt | cut -c1-8) equity=$(md5sum hearts_equity.pt | cut -c1-8) units=${#UNITS[@]} conc=$CONC" >> "$LOG"
lowpri() {
  [ "${HEARTS_NO_LOWPRI:-0}" = "1" ] && return
  powershell -NoProfile -Command "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" > /dev/null 2>&1
}
run_unit() {
  local name=$1 seed=$2 rc
  rm -f "$OUT/$name.csv" "$OUT/$name.tricks.csv" "$OUT/$name.done"
  "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --opponent-model "$TRACE" --shooter sel --pass-search --k 64 --matches 1 \
    --seed "$seed" --cuda --out "$OUT/$name.csv" --tricks-out "$OUT/$name.tricks.csv" \
    > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$OUT/$name.csv" ]; then echo "$seed $(date +%s)" > "$OUT/$name.done"; fi
  echo "unit $name DONE rc=$rc $(date +%T)" >> "$LOG"
  return $rc
}
fail=0; started=0
for entry in "${UNITS[@]}"; do
  name=${entry%%:*}; seed=${entry##*:}
  [ -f "$OUT/$name.done" ] && continue
  [ -f r9_STOP ] && { echo "R11_STOP file seen; not starting more units $(date)" >> "$LOG"; break; }
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n || fail=1; done
  run_unit "$name" "$seed" &
  started=$((started + 1))
  sleep 2; lowpri
done
wait || fail=1
missing=0
for entry in "${UNITS[@]}"; do name=${entry%%:*}; [ -f "$OUT/$name.done" ] || missing=$((missing + 1)); done
echo "R11_END mode=$MODE started=$started missing=$missing $(date)" >> "$LOG"
if [ "$missing" = "0" ]; then echo "R11_DONE mode=$MODE $(date)" >> "$LOG"; exit 0; fi
echo "R11_INCOMPLETE mode=$MODE missing=$missing (re-run to resume) $(date)" >> "$LOG"; exit 1
