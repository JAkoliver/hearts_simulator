"""Phase B distillation: recorded search-shooter decisions -> a small fast
policy net (docs/exploiter_league_prereg.md).

The distilled nets are TRAINING OPPONENTS for the exploiter league, never
candidates and never gates: match-mode PPO needs millisecond opponents
that carry a real moon threat, and the search-shooter runs at search
speed. Imitation is hard-CE on the shooter's own choices (play AND pass
picks - moon-keeping passes are half the threat).

Record format (SearchEval --record-out, 2284 B/decision, little-endian):
    float32 obs[556] | uint8 mask[52] | int32 action | uint8 flags | pad[3]
    flags: bit0 pass_phase, bit1 moon-alive, bit2 shooting

Holdout is by SHARD (a shard is a disjoint match-seed block), so no
same-match leakage inflates the metric.

Usage:
  python train_shooter.py --mode agg --out shooter_agg_v1.pth
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hearts_net import HeartsNetV5

REC = 556 * 4 + 52 + 4 + 1 + 3


def load_shard(path):
    buf = np.fromfile(path, dtype=np.uint8)
    n = len(buf) // REC
    a = buf[:n * REC].reshape(n, REC)
    obs = a[:, :556 * 4].view(np.float32).reshape(n, 556).copy()
    mask = a[:, 556 * 4:556 * 4 + 52].astype(bool)
    act = a[:, 556 * 4 + 52:556 * 4 + 56].copy().view(np.int32).ravel()
    flags = a[:, 556 * 4 + 56]
    if not mask[np.arange(n), act].all():
        raise SystemExit(f'{path}: recorded action outside the legal mask')
    return obs, mask, act.astype(np.int64), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='agg', choices=('agg', 'sel'))
    ap.add_argument('--data', default='expert_data/shooter_v1')
    ap.add_argument('--out', default=None)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d-model', type=int, default=192)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=707)
    args = ap.parse_args()
    out = args.out or f'shooter_{args.mode}_v1.pth'
    torch.manual_seed(args.seed)

    paths = sorted(p for p in glob.glob(
        os.path.join(args.data, f'phaseB_{args.mode}_*.sdrec')))
    if len(paths) < 2:
        raise SystemExit(f'need >=2 shards for a by-shard holdout, found {len(paths)}')
    val_path, train_paths = paths[-1], paths[:-1]
    print(f'{args.mode}: train shards {[os.path.basename(p) for p in train_paths]}'
          f' | holdout {os.path.basename(val_path)}')

    tr = [load_shard(p) for p in train_paths]
    obs = torch.from_numpy(np.concatenate([t[0] for t in tr]))
    mask = torch.from_numpy(np.concatenate([t[1] for t in tr]))
    act = torch.from_numpy(np.concatenate([t[2] for t in tr]))
    flags = np.concatenate([t[3] for t in tr])
    vo, vm, va, vf = load_shard(val_path)
    vo = torch.from_numpy(vo); vm = torch.from_numpy(vm)
    va = torch.from_numpy(va)
    print(f'  train {len(act):,} decisions ({(flags & 1).mean():.1%} pass) '
          f'| holdout {len(va):,}')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = HeartsNetV5(obs_dim=556, d_model=args.d_model,
                      num_layers=args.layers, num_heads=args.heads).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f'  net d={args.d_model} L={args.layers} ({n_par/1e6:.2f}M params) on {dev}')
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs *
                                             (len(act) // args.batch + 1)))

    def evaluate(o, m, a, chunk=4096):
        net.eval()
        hits = 0
        with torch.no_grad():
            for i in range(0, len(a), chunk):
                lg, _ = net(o[i:i+chunk].to(dev), m[i:i+chunk].to(dev))
                hits += (lg.argmax(1).cpu() == a[i:i+chunk]).sum().item()
        net.train()
        return hits / len(a)

    t0 = time.time()
    best = 0.0
    for ep in range(args.epochs):
        perm = torch.randperm(len(act))
        tot = nb = 0
        for i in range(0, len(act) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            lg, _ = net(obs[idx].to(dev), mask[idx].to(dev))
            loss = F.cross_entropy(lg, act[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        acc = evaluate(vo, vm, va)
        print(f'  epoch {ep+1}/{args.epochs}: loss {tot/max(1,nb):.4f} '
              f'| holdout match {acc:.4f} ({time.time()-t0:.0f}s)')
        if acc > best:
            best = acc
            torch.save({'state_dict': net.state_dict(), 'd_model': args.d_model,
                        'num_layers': args.layers, 'num_heads': args.heads,
                        'mode': args.mode,
                        'holdout_match': acc, 'n_train': int(len(act)),
                        'shards': [os.path.basename(p) for p in train_paths]},
                       out)
    with open(out, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()[:12]
    print(f'saved {out} (md5 {md5}) best holdout match {best:.4f}')
    json.dump({'mode': args.mode, 'out': out, 'md5': md5,
               'holdout_match': best, 'n_train': int(len(act)),
               'params_m': round(n_par / 1e6, 3)},
              open(f'equity_data/verdicts/shooter_{args.mode}_v1_distill.json', 'w'),
              indent=1)


if __name__ == '__main__':
    main()
