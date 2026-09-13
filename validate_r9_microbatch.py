"""League r9 null contract (d): always-on micro-batching is equivalent.

One PPO update on a FIXED synthetic buffer (real obs-v2 states from the v6
holdout, random legal actions, synthetic rewards/values), run twice from
the same initial weights with identical RNG: micro_batch = full minibatch
vs micro_batch = 512. Registered bar: max |delta param| <= 1e-5 (the
measured 8e-6 order of the original headroom-mode equivalence). Exit 1 on
failure.
"""
import copy
import random
import sys

import numpy as np
import torch
import torch.optim as optim

from hearts_net import net_from_checkpoint
from train import RolloutBuffer, ppo_update
from v6_probe_eval import walk_holdout

ARMA = 'v6_stage3/arma_lr1e-4.ep3.pth'
N = 3000
MB = 2048
BAR = 1e-5


def make_buffer(n, seed=7):
    hold, _ = walk_holdout()
    obs = np.concatenate([np.ascontiguousarray(hold['obs']),
                          np.ascontiguousarray(hold['ext'])], axis=1)[:n].astype(np.float32) / 255.0
    mask = np.ascontiguousarray(hold['mask'])[:n].astype(bool)
    rng = np.random.default_rng(seed)
    b = RolloutBuffer()
    for i in range(n):
        legal = np.flatnonzero(mask[i])
        b.states.append(obs[i])
        b.actions.append(int(rng.choice(legal)))
        b.log_probs.append(float(-np.log(len(legal))))
        b.rewards.append(float(rng.normal(0, 1)))
        b.values.append(float(rng.normal(0, 0.5)))
        b.masks.append(mask[i])
        b.dones.append(bool((i + 1) % 20 == 0))
        b.hand_labels.append(np.zeros(156, dtype=np.float32))
    return b


def one_update(micro, buf, device):
    torch.manual_seed(1234); random.seed(1234); np.random.seed(1234)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(1234)
    net = net_from_checkpoint(ARMA).to(device)
    opt = optim.Adam(net.parameters(), lr=5e-5)
    ppo_update(net, opt, buf, device, gamma=1.0, eps_clip=0.16, k_epochs=1,
               minibatch_size=MB, gae_lambda=0.9, entropy_coef=0.002,
               aux_coef=0.77, micro_batch=micro)
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    buf = make_buffer(N)
    full = one_update(MB, buf, device)      # micro == minibatch: the r8 full-speed path
    micro = one_update(512, buf, device)    # the r9 always-on path
    init = net_from_checkpoint(ARMA).state_dict()
    moved = max(float((full[k] - init[k].float()).abs().max()) for k in full)
    diff = max(float((full[k] - micro[k]).abs().max()) for k in full)
    print(f'params moved by the update (max |d|): {moved:.3e}')
    print(f'full-minibatch vs micro-batched 512: max |delta param| = {diff:.3e} (bar {BAR:.0e})')
    if moved < 1e-7:
        print('FAIL: update moved nothing - test is vacuous'); sys.exit(1)
    if diff > BAR:
        print('R9 CONTRACT (d) FAIL'); sys.exit(1)
    print('R9 CONTRACT (d) PASS: micro-batching equivalent within bar')
