"""League r11 exposure census (docs/exploiter_league_r11_prereg.md §3.1).

Self-play of the r10 B1 candidate on the vec harness (all four seats, no
shooter). On every decision where the PROMOTED gate fires (router max-
opponent moon prob p > 0.1 AND an opponent moon-alive flag) record p
ONLY. Output: quantiles that fix the round's arms, the histogram, and the
gated share of all decisions. OUTCOME-BLIND by construction: no score,
moon or placement statistic is computed or written.

    python r11_exposure_census.py --matches 1000 --seed 770000000
"""
import argparse
import json
import time

import numpy as np
import torch

import headroom
from hearts_match_env import MatchVecEnv
from hearts_net import net_from_checkpoint

B1_CAND = 'equity_data/exploiter_r9/B1/r9_B1_end_ensemble.pth'   # d5222b1d
T_LO = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=770_000_000)
    ap.add_argument('--json', default='equity_data/exploiter_r11/census.json')
    ap.add_argument('--bs', type=int, default=2048)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net_from_checkpoint(B1_CAND).eval().to(device)
    assert net.gate == 'moonhead:0.1' and net.router is not None
    n = args.matches
    vec = MatchVecEnv(n, args.seed)
    alive = np.ones(n, dtype=bool)
    ps, total, t0, steps = [], 0, time.time(), 0
    while alive.any():
        headroom.pace()
        g = np.flatnonzero(alive)
        obs_np = vec.observe_v2_batch(g)
        mask_np = vec.legal_mask_batch(g)
        actions = np.zeros(len(g), dtype=np.int64)
        for i in range(0, len(g), args.bs):
            obs = torch.from_numpy(obs_np[i:i + args.bs]).to(device)
            mask = torch.from_numpy(mask_np[i:i + args.bs]).to(device)
            with torch.no_grad():
                logits, _ = net(obs, mask)                    # fp32, as served
                _, _, _, moon, _ = net.router.forward_aux(obs, mask)
                p = torch.sigmoid(moon[:, 1:]).max(dim=1).values
                al = (obs[:, 872:876] > 0.5)[:, 1:].any(dim=1)
                fired = (p > T_LO) & al
            actions[i:i + args.bs] = logits.argmax(1).cpu().numpy()
            ps.append(p[fired].float().cpu().numpy())
        total += len(g)
        _, match_dones, _, _ = vec.step_batch(g, actions)
        for j in np.flatnonzero(match_dones):
            alive[int(g[j])] = False
        steps += 1
        if steps % 500 == 0:
            print(f'  step {steps}: {int((~alive).sum())}/{n} matches done '
                  f'({time.time() - t0:.0f}s)', flush=True)

    p = np.concatenate(ps)
    q25, q50, q75 = (float(np.percentile(p, q)) for q in (25, 50, 75))
    hist, edges = np.histogram(p, bins=np.linspace(0.1, 1.0, 19))
    out = {'matches': n, 'seed': args.seed, 'decisions': int(total),
           'gated': int(len(p)), 'gated_share': len(p) / total,
           'quantiles': {'q25': q25, 'q50': q50, 'q75': q75},
           'arms': {'E75': round(q25, 2), 'E50': round(q50, 2), 'E25': round(q75, 2)},
           'hist_edges': [round(float(e), 3) for e in edges],
           'hist_counts': [int(c) for c in hist],
           'seconds': time.time() - t0}
    with open(args.json, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
