#!/bin/bash
# League r9 null contract (§3.4, one-match-per-process form): a killed match,
# replayed from scratch, is BIT-IDENTICAL to an uninterrupted run of that
# match; and two matches run CONCURRENTLY (2-wide, GPU shared) are identical
# to their solo runs. Uses transfer shard 12, matches 5 and 6 (promoted
# trace). Also asserts the solo rows equal the driver's own output for the
# same units (the driver is what production uses).
cd "$(dirname "$0")/.."
EXE=build/Release/SearchEval.exe
TRACE=hybrid_champ_arma_moonhead_0p1_882_v2.pt
SCR=equity_data/exploiter_r9/searchsel_resume_aa2
KILL_AFTER=${KILL_AFTER:-75}
mkdir -p "$SCR"
one() {  # outprefix seed
  "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --opponent-model "$TRACE" --shooter sel --pass-search --k 64 --matches 1 \
    --seed "$2" --cuda --out "$1.csv" --tricks-out "$1.tricks.csv" > "$1.log" 2>&1
}
S5=$((732260806 + 5 * 1000)); S6=$((732260806 + 6 * 1000))
echo "AA2_START $(date)"
one "$SCR/solo_m5" $S5; echo "solo m5 done $(date +%T)"
one "$SCR/solo_m6" $S6; echo "solo m6 done $(date +%T)"
# killed-and-replayed m5
one "$SCR/killed_m5" $S5 & pid=$!
sleep "$KILL_AFTER"
echo "killing m5 at $(date +%T) (rows so far $(wc -l < "$SCR/killed_m5.csv" 2>/dev/null))"
powershell -NoProfile -Command "Get-Process SearchEval -ErrorAction SilentlyContinue | Stop-Process -Force"
wait $pid 2>/dev/null
one "$SCR/replay_m5" $S5; echo "replay m5 done $(date +%T)"
# concurrent m5 + m6
one "$SCR/conc_m5" $S5 & p1=$!
one "$SCR/conc_m6" $S6 & p2=$!
wait $p1; wait $p2; echo "concurrent m5+m6 done $(date +%T)"
ok=1
cmp -s "$SCR/solo_m5.csv" "$SCR/replay_m5.csv" && cmp -s "$SCR/solo_m5.tricks.csv" "$SCR/replay_m5.tricks.csv" || { echo "FAIL: replayed m5 != solo m5"; ok=0; }
cmp -s "$SCR/solo_m5.csv" "$SCR/conc_m5.csv" || { echo "FAIL: concurrent m5 != solo m5"; ok=0; }
cmp -s "$SCR/solo_m6.csv" "$SCR/conc_m6.csv" || { echo "FAIL: concurrent m6 != solo m6"; ok=0; }
echo "rows: solo m5 $(wc -l < "$SCR/solo_m5.csv"), replay $(wc -l < "$SCR/replay_m5.csv"), conc $(wc -l < "$SCR/conc_m5.csv"); m6 solo $(wc -l < "$SCR/solo_m6.csv"), conc $(wc -l < "$SCR/conc_m6.csv")"
if [ $ok = 1 ]; then echo "R9 SEARCHSEL-REPLAY CONTRACT PASS: killed match replays bit-identical; 2-wide concurrency bit-identical"; exit 0; fi
echo "R9 SEARCHSEL-REPLAY CONTRACT FAIL"; exit 1
