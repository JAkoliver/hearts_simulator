#!/bin/bash
# B2 drift-screen HALT EXPLORATION (08-09): holdout-only probes to pick
# the ONE registered amendment. Pattern = expert-iter v2 freeze
# exploration. Candidates here are NEVER gate-eligible.
cd "$(dirname "$0")/.."
set -e
run() { echo "B2EXPLORE $* $(date)";
  PYTHONUNBUFFERED=1 python -u train_b2.py --exploration --epochs 3 "$@" 2>&1; }
# A: budget-epoch (share = true dose control), share 0.95
run --anchor-share 0.95 --lr 1e-5 --epoch-budget 165000 --out cand_b2ex_budget_s95.pth
# B: dose via LR (current epoch definition)
run --anchor-share 0.875 --lr 3e-6 --out cand_b2ex_lr3e6.pth
# C: KL-to-distribution anchor, two weights
run --anchor-share 0.875 --lr 1e-5 --anchor-loss kl --kl-coef 1.0 --out cand_b2ex_kl1.pth
run --anchor-share 0.875 --lr 1e-5 --anchor-loss kl --kl-coef 4.0 --out cand_b2ex_kl4.pth
echo "B2EXPLORE_COMPLETE $(date)"
