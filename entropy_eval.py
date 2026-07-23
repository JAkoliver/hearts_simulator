"""Mean policy entropy of a checkpoint over sampled bank observations.

Used by the sharpen-sweep selection rule (2026-07-23): over-sharpened
distillation targets can collapse policy entropy and hurt subsequent PPO
exploration, so within-noise gate candidates are broken toward higher
entropy.

Usage: python entropy_eval.py --model <ckpt> [--data glob ...] [--n 8192]
"""
import argparse
import glob

import numpy as np
import torch

from distill import RECORD
from hearts_net import net_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', nargs='+',
                    default=['selfplay_data/0722_fresh_iter0/*.bin'])
    ap.add_argument('--n', type=int, default=8192)
    args = ap.parse_args()

    files = sorted(f for p in args.data for f in glob.glob(p))
    if not files:
        raise SystemExit('no data files matched')
    chunks, need = [], args.n
    for f in files:
        arr = np.fromfile(f, dtype=RECORD)
        chunks.append(arr[:need])
        need -= len(chunks[-1])
        if need <= 0:
            break
    recs = np.concatenate(chunks)

    net = net_from_checkpoint(args.model)
    net.eval()
    obs = torch.from_numpy(np.ascontiguousarray(recs['obs'])).float() / 255.0
    mask = torch.from_numpy(np.ascontiguousarray(recs['mask'])).bool()

    ents, counts = [], []
    with torch.no_grad():
        for i in range(0, len(recs), 2048):
            logits, _ = net(obs[i:i + 2048], mask[i:i + 2048])
            p = torch.softmax(logits, dim=1)
            logp = torch.log(p.clamp_min(1e-12))
            ent = -(p * logp).masked_fill(~mask[i:i + 2048], 0.0).sum(dim=1)
            ents.append(ent)
            counts.append(mask[i:i + 2048].sum(dim=1).float())
    ents = torch.cat(ents)
    counts = torch.cat(counts)
    print(f"{args.model}: mean entropy {ents.mean().item():.4f} nats "
          f"(n={len(ents)}, mean legal moves {counts.mean().item():.2f}, "
          f"max possible ~{np.log(counts.mean().item()):.2f})")


if __name__ == '__main__':
    main()
