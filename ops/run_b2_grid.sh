#!/bin/bash
# Round-2 Phase B2 registered candidate grid (prereg: anchor share x LR,
# <=3 epochs, at most 4 configs). Sequential; per-config logs.
cd "$(dirname "$0")/.."
set -e
for cfg in "0.75 1e-5 s75_lr1" "0.75 3e-5 s75_lr3" \
           "0.875 1e-5 s875_lr1" "0.875 3e-5 s875_lr3"; do
  set -- $cfg
  echo "B2GRID config share=$1 lr=$2 tag=$3 $(date)"
  PYTHONUNBUFFERED=1 python -u train_b2.py --anchor-share "$1" --lr "$2" \
    --epochs 3 --out "cand_b2_$3.pth" 2>&1
done
echo "B2GRID_COMPLETE $(date)"
