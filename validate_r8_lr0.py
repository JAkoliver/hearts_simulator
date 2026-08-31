"""Null contract (a) checker for league r8: after a SMOKE_TEST run with
learning_rate 0 in ensemble-learner mode, hearts_model_final.pth must be
BIT-IDENTICAL to the specialist init (Adam with lr=0 moves nothing; any
difference means something other than the optimizer touched the learner).
Run: SMOKE_TEST=1 python train.py  (config = config_r8_nullA.json), then
this. Exit 1 on any differing tensor."""
import sys

import torch

INIT = 'v6_stage3/arma_lr1e-4.ep3.pth'
TRAINED = 'hearts_model_final.pth'

if __name__ == '__main__':
    a = torch.load(INIT, weights_only=True, map_location='cpu')
    b = torch.load(TRAINED, weights_only=True, map_location='cpu')
    if set(a) != set(b):
        print(f'key sets differ: only-init {set(a) - set(b)}, '
              f'only-trained {set(b) - set(a)}')
        sys.exit(1)
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    if bad:
        print(f'{len(bad)} tensors differ, e.g. {bad[:5]}')
        sys.exit(1)
    print(f'lr-0 NULL PASS: all {len(a)} tensors bit-identical to the init')
