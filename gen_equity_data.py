"""Equity-model training data generator (docs/match_aware_search_design.md).

Plays matches to 100 with the CURRENT baseline raw net at ALL FOUR seats
(matching the rollout-leaf context where the equity model will be used)
and records every deal-boundary state -> final placements.

Seeding (training mode, per the design's coverage mixture):
  50% uniform totals on [0,99]^4; 30% one seat 85-99 (threshold tails);
  10% two seats >=85 within 10 (near-threshold ties); 10% natural (0s).
deals_played is derived from totals (sum/26 + noise) and the CURRENT
deal's pass direction is aligned via deals_played mod 4 extra Reset()s
(verified: Reset alone advances the rotation).

--mode natural generates the DEDICATED calibration holdout: all matches
from zero, natural trajectories only, kept separate from training.

Output: one .npz per run: totals (n,4 raw seat order), deals (n,),
pass_dir (n,), placements (n,4), match_id (n,), mixture (n,).

Usage:
  python gen_equity_data.py --matches 30000 --mode seeded \
      --out equity_data/train_seeded_v1.npz [--workers 10] [--seed 1]
"""
import argparse
import os
import time

import numpy as np
import torch

import headroom
from hearts_match_env import MatchEnv
from hearts_net import net_from_checkpoint

_SEED_STRIDE = 1_000_000


def _sample_start(rng, mode):
    """Returns (totals[4] float, deals_played int, mixture_tag int)."""
    if mode == 'natural':
        return np.zeros(4), 0, 3
    u = rng.random()
    if u < 0.5:
        totals = rng.integers(0, 100, 4).astype(np.float64)
        tag = 0
    elif u < 0.8:
        totals = rng.integers(0, 85, 4).astype(np.float64)
        totals[rng.integers(0, 4)] = rng.integers(85, 100)
        tag = 1
    elif u < 0.9:
        totals = rng.integers(0, 85, 4).astype(np.float64)
        a, b = rng.choice(4, size=2, replace=False)
        hi = rng.integers(85, 100)
        totals[a] = hi
        totals[b] = np.clip(hi + rng.integers(-9, 10), 85, 99)
        tag = 2
    else:
        return np.zeros(4), 0, 3
    deals = max(0, int(round(totals.sum() / 26.0)) + int(rng.integers(-1, 2)))
    return totals, deals, tag


def _chunk(job):
    ckpt, mode, seed, offset, n_matches = job
    torch.set_num_threads(1)
    headroom.apply_process_priority()
    net = net_from_checkpoint(ckpt)
    net.eval()
    rng = np.random.default_rng(seed)

    rows = {k: [] for k in ('totals', 'deals', 'pass_dir', 'placements',
                            'match_id', 'mixture')}

    def record(menv, mid, tag, placements=None):
        rows['totals'].append(menv.match_scores.copy())
        rows['deals'].append(menv.deals_played)
        rows['pass_dir'].append(int(menv.env.get_pass_direction()))
        rows['match_id'].append(mid)
        rows['mixture'].append(tag)
        rows['placements'].append(placements)  # filled at match end

    for m in range(n_matches):
        mid = offset + m
        totals, deals, tag = _sample_start(rng, mode)
        menv = MatchEnv(seed=int(seed + m * 7 + 1))
        for _ in range(deals % 4):
            menv.env.reset()  # align pass rotation with deals_played
        menv.match_scores = totals.copy()
        menv.deals_played = deals

        boundary_indices = [len(rows['totals'])]
        record(menv, mid, tag)
        while True:
            obs = torch.from_numpy(menv.observe()).unsqueeze(0)
            mask = torch.zeros((1, 52), dtype=torch.bool)
            for a in menv.get_legal_actions():
                if a != -1:
                    mask[0, a] = True
            with torch.no_grad():
                logits, _ = net(obs, mask)
            deal_done, match_done, _ = menv.step(int(logits.argmax(1).item()))
            if match_done:
                pl = menv.placements()
                for bi in boundary_indices:
                    rows['placements'][bi] = pl.copy()
                break
            if deal_done:
                boundary_indices.append(len(rows['totals']))
                record(menv, mid, tag)
            if menv.deals_played > 60:
                raise RuntimeError('runaway match')
    return {k: np.array(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', type=int, required=True)
    ap.add_argument('--mode', choices=['seeded', 'natural'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='hearts_model_final.pth')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    headroom.banner()
    per = args.matches // args.workers
    extra = args.matches % args.workers
    jobs, offset = [], 0
    for w in range(args.workers):
        n = per + (1 if w < extra else 0)
        if n == 0:
            continue
        jobs.append((args.ckpt, args.mode, args.seed + w * _SEED_STRIDE, offset, n))
        offset += n

    import multiprocessing
    t0 = time.time()
    with multiprocessing.Pool(len(jobs),
                              initializer=headroom.apply_process_priority) as pool:
        parts = pool.map(_chunk, jobs)

    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, **merged,
                        ckpt=args.ckpt, mode=args.mode, seed=args.seed)
    n = len(merged['totals'])
    el = time.time() - t0
    print(f"{args.mode}: {args.matches} matches -> {n} boundary states "
          f"in {el:.0f}s ({args.matches / el * 3600:.0f} matches/h). "
          f"mixture counts: {np.bincount(merged['mixture'], minlength=4).tolist()}")
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
