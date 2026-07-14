"""Supervised distillation trainer for expert iteration.

Trains the network to imitate the search teacher's action-value distribution
(policy head, soft-target cross-entropy), predict round outcomes (value head,
MSE), and predict hidden hands (belief head, BCE) from SelfPlayGen binary
records. Warm-starts from the
current baseline so each iteration refines rather than restarts.

Usage:
  python distill.py --data "selfplay_data/iter_0/*.bin" \
      --init hearts_model_final.pth --out hearts_model_candidate.pth --epochs 2
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hearts_net import HeartsNet

RECORD = np.dtype([
    ('obs', 'u1', 550), ('mask', 'u1', 52), ('labels', 'u1', 156),
    ('pi', 'u1', 52),
    ('action', '<u2'), ('seat', '<u2'), ('reward', '<f4'),
])
assert RECORD.itemsize == 818

def load_data(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise SystemExit(f"No data files match {patterns}")
    for f in sorted(files):
        if os.path.getsize(f) % RECORD.itemsize != 0:
            raise SystemExit(
                f"{f}: size is not a multiple of {RECORD.itemsize} - generated "
                f"by an older SelfPlayGen (766-byte records)? Regenerate the data.")
    chunks = [np.fromfile(f, dtype=RECORD) for f in sorted(files)]
    data = np.concatenate(chunks)
    print(f"Loaded {len(data):,} records from {len(files)} files")
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--init', default='hearts_model_final.pth')
    ap.add_argument('--out', default='hearts_model_candidate.pth')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--value-coef', type=float, default=0.5)
    ap.add_argument('--aux-coef', type=float, default=0.5)
    args = ap.parse_args()

    data = load_data(args.data)

    net = HeartsNet()
    if args.init and os.path.exists(args.init):
        net.load_state_dict(torch.load(args.init, weights_only=True))
        print(f"Warm start from {args.init}")
    else:
        print("Fresh network (no init checkpoint found)")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    n = len(data)
    reward_var = float(np.var(data['reward'])) + 1e-8

    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        ce_sum = match_sum = err2_sum = bce_sum = 0.0
        seen = 0

        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            b = data[idx]
            obs = torch.from_numpy(b['obs'].astype(np.float32) / 255.0)
            mask = torch.from_numpy(b['mask'].astype(bool))
            actions = torch.from_numpy(b['action'].astype(np.int64))
            labels = torch.from_numpy(b['labels'].astype(np.float32))
            rewards = torch.from_numpy(b['reward'].astype(np.float32))
            pi = torch.from_numpy(b['pi'].astype(np.float32))
            pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-6)

            logits, value, belief = net.forward_all(obs, mask)
            # Soft-target cross-entropy against the teacher's value-derived
            # distribution. Illegal logits are -inf; zero them in log-space
            # (their pi is 0) so 0 * -inf can't poison the loss with NaN.
            logp = F.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
            policy_loss = -(pi * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), rewards)
            belief_loss = F.binary_cross_entropy_with_logits(belief, labels)
            loss = policy_loss + args.value_coef * value_loss + args.aux_coef * belief_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            k = len(idx)
            seen += k
            ce_sum += policy_loss.item() * k
            match_sum += (logits.argmax(1) == actions).float().sum().item()
            err2_sum += value_loss.item() * k
            bce_sum += belief_loss.item() * k

        ev = 1.0 - (err2_sum / seen) / reward_var
        print(f"epoch {epoch + 1}/{args.epochs} | policy CE {ce_sum / seen:.4f} | "
              f"teacher match {match_sum / seen * 100:.1f}% | value EV {ev:.3f} | "
              f"belief BCE {bce_sum / seen:.4f}")

    torch.save(net.state_dict(), args.out)
    print(f"Saved candidate to {args.out}")

if __name__ == '__main__':
    main()
