#!/bin/bash
# Phase 2 Stage A verification: A/A determinism + recorder checks.
cd "$(dirname "$0")/.."
set -e
mkdir -p expert_data/p2 logs
for run in a b; do
  build/Release/SearchEval --tree-selfplay \
    --search-model hearts_ai_search_match.pt --equity-model hearts_equity.pt \
    --iterations 64 --k 24 --matches 1 --seed 210000000 --cuda \
    --record-out expert_data/p2/aa_$run.hvt \
    > logs/p2_aa_$run.log 2>&1
  echo "AA run $run done rc=$?"
done
md5sum expert_data/p2/aa_a.hvt expert_data/p2/aa_b.hvt
