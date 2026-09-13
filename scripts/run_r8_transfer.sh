#!/bin/bash
# League r8 TRANSFER CHECK (docs/exploiter_league_r8_prereg.md §4): chunked
# search-SEL n=64 for a candidate ensemble trace — shards 0,1 of block
# 720260806 (32 matches each, K=64 flat, pass search, SEL search shooter),
# the r7 flag set verbatim; paired post-hoc against the PROMOTED ensemble's
# r7 rows (equity_data/exploiter_r4/r7_gate/cand_{0,1}.csv) and the
# champion base rows. Registered GATE-FIRES pre-check first (4 matches on
# seed 720260806 must differ from BOTH the champion and the promoted rows).
# Resumable: a shard with a complete CSV (33 header+... rows checked by
# the analyzer) is skipped. <= 2 concurrent SearchEval, BelowNormal.
#   bash scripts/run_r8_transfer.sh <tag> <trace_882.pt>
cd "$(dirname "$0")/.."
TAG=${1:?tag}; TRACE=${2:?trace}
OUT=equity_data/exploiter_r8/$TAG/searchsel; LOG=$OUT/driver.log
mkdir -p "$OUT"
EXE=build/Release/SearchEval.exe
echo "R8_TRANSFER_START $(date) tag=$TAG trace=$(md5sum "$TRACE" | cut -c1-8) exe=$(md5sum $EXE | cut -c1-8) search=$(md5sum hearts_ai_search_match.pt | cut -c1-8) equity=$(md5sum hearts_equity.pt | cut -c1-8)" >> "$LOG"
lowpri() { sleep 3; powershell -NoProfile -Command "Get-Process SearchEval -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'BelowNormal' }" > /dev/null 2>&1; }

if [ ! -s "$OUT/gatefires_4m.csv" ]; then
  "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --opponent-model "$TRACE" --shooter sel --pass-search --k 64 --matches 4 \
    --seed 720260806 --cuda --out "$OUT/gatefires_4m.csv" > "$OUT/gatefires.log" 2>&1 &
  lowpri; wait
fi
head -5 "$OUT/gatefires_4m.csv" > "$OUT/.gf"
head -5 equity_data/exploiter_r4/r7_gate/base_0.csv > "$OUT/.champ"
head -5 equity_data/exploiter_r4/r7_gate/cand_0.csv > "$OUT/.promoted"
if cmp -s "$OUT/.gf" "$OUT/.champ"; then echo "R8_HALT gate-fires FAILED: identical to CHAMPION rows $(date)" >> "$LOG"; exit 1; fi
if cmp -s "$OUT/.gf" "$OUT/.promoted"; then echo "R8_HALT gate-fires FAILED: identical to PROMOTED rows $(date)" >> "$LOG"; exit 1; fi
echo "R8 gate-fires PASS (differs from champion AND promoted) $(date)" >> "$LOG"

pids=()
for i in 0 1; do
  if [ -s "$OUT/cand_$i.csv" ] && [ "$(cut -d, -f1 "$OUT/cand_$i.csv" | sort -u | wc -l)" -ge 33 ]; then
    echo "shard $i already complete, skipping $(date)" >> "$LOG"; continue
  fi
  "$EXE" --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --opponent-model "$TRACE" --shooter sel --pass-search \
    --k 64 --matches 32 --seed $((720260806 + i * 1000000)) --cuda \
    --out "$OUT/cand_$i.csv" --tricks-out "$OUT/cand_$i.tricks.csv" \
    > "$OUT/shard_$i.log" 2>&1 &
  pids+=($!); echo "shard $i START seed $((720260806 + i * 1000000)) $(date)" >> "$LOG"
done
lowpri
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "R8_TRANSFER_DONE rc=$rc $(date)" >> "$LOG"
