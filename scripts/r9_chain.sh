#!/bin/bash
# League r9 trial chain (docs/exploiter_league_r9_prereg.md §4/§10): A1 -> B1 -> A2 -> B2,
# each trial followed by its readout (vec probe both clones -> fail-fast check ->
# strength -> transfer n=128) and the readout analysis. Every stage goes through
# r9_pipeline.py (preflight + r9_state.json bookkeeping), so the chain is resumable:
# re-running it skips completed units and resumes an interrupted trial from its
# checkpoint. HALT-DEFAULT: any non-zero stage stops the chain.
#   bash scripts/r9_chain.sh "full speed"        (or "headroom 0.25")
cd "$(dirname "$0")/.."
MODE=${1:?mode}
LOG=equity_data/exploiter_r9/chain.log
echo "CHAIN_START $(date) mode=$MODE" >> "$LOG"
for T in A1 B1 A2 B2; do
  for STAGE in "trial $T" "readout $T"; do
    echo "STAGE $STAGE START $(date)" >> "$LOG"
    PYTHONUNBUFFERED=1 python -u scripts/r9_pipeline.py $STAGE --mode "$MODE" >> "equity_data/exploiter_r9/chain_${T}.log" 2>&1
    rc=$?
    echo "STAGE $STAGE DONE rc=$rc $(date)" >> "$LOG"
    if [ $rc -ne 0 ]; then echo "CHAIN_HALT at $STAGE rc=$rc $(date)" >> "$LOG"; exit $rc; fi
  done
  python analyze_r9_readout.py "$T" >> "equity_data/exploiter_r9/chain_${T}.log" 2>&1
  echo "READOUT $T: $(python analyze_r9_readout.py "$T" 2>/dev/null | tail -1)" >> "$LOG"
done
echo "CHAIN_DONE $(date)" >> "$LOG"
