#!/bin/bash
# Match-aware expert-iteration distill (pre-registered recipe):
# warm-start from champion 8a89da90, sharpen 2.0 (measured optimum),
# 3 epochs (value head overfits from ~4), 10% by-match tail holdout.
cd /e/hearts_simulator || exit 1
export PYTHONUNBUFFERED=1
python -u distill.py --match \
  --data expert_data/matchgen_natural_t*.bin expert_data/matchgen2_*.bin \
         expert_data/fast_*.bin expert_data/gentle_*.bin \
         expert_data/day2_*.bin expert_data/fastday_*.bin \
  --init hearts_model_final.pth \
  --out cand_expert_iter1.pth \
  --epochs 3 --sharpen 2.0 --holdout 0.10 --device cuda
echo "DISTILL_RC=$?"
