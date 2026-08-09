"""Generate the landing-page attract-mode demo matches (static JSON).

Plays all-AI matches with the SITE's served weights (hearts_web_model.pth
= promoted baseline; the demo shows the real product), records a render
script per match - plays with the net's top-3 policy preview (the
review-attract's bars), trick winners, per-deal scores - and bakes the
best three to hearts_web/static/demo/match_{0,1,2}.json.

Selection: from 8 candidate seeds, prefer matches containing a shot
moon (best five-second television), then shorter matches (file size,
loop cadence). Pure CPU - safe to run while GPU work is in flight.

Run from repo root: python hearts_web/gen_demo_matches.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hearts_match_env import MatchEnv  # noqa: E402
from hearts_net import net_from_checkpoint  # noqa: E402

SUITS = 'CDSH'
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'static', 'demo')
CAND_SEEDS = [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108]


def card_name(idx):
    return RANKS[idx % 13] + SUITS[idx // 13]


def play_match(net, seed):
    menv = MatchEnv(seed=seed)
    deals = []
    cur = {'pass_dir': None, 'plays': [], 'winners': []}
    trick = []
    has_moon = False
    while True:
        obs = torch.from_numpy(menv.observe()).unsqueeze(0)
        mask = torch.zeros((1, 52), dtype=torch.bool)
        for a in menv.get_legal_actions():
            if a != -1:
                mask[0, a] = True
        with torch.no_grad():
            logits, _, _ = net.forward_all(obs, mask)
        probs = torch.softmax(logits[0], dim=0)
        action = int(torch.argmax(probs).item())
        seat = menv.get_current_player()
        if menv.is_passing():
            if cur['pass_dir'] is None:
                cur['pass_dir'] = ['left', 'right', 'across', 'hold'][
                    int(menv.env.get_pass_direction())]
        else:
            if cur['pass_dir'] is None:
                cur['pass_dir'] = ['left', 'right', 'across', 'hold'][
                    int(menv.env.get_pass_direction())]
            top = torch.topk(probs, k=min(3, int(mask.sum().item())))
            cur['plays'].append(
                {'s': seat, 'c': card_name(action),
                 't3': [[card_name(int(i)), round(float(p), 3)]
                        for p, i in zip(top.values, top.indices)]})
            trick.append((seat, action))
            if len(trick) == 4:
                lead = trick[0][1] // 13
                w = max((c for c in trick if c[1] // 13 == lead),
                        key=lambda c: c[1] % 13)[0]
                cur['winners'].append(w)
                trick = []
        deal_done, match_done, rs = menv.step(action)
        if deal_done:
            rs = list(map(int, rs))
            srt = sorted(rs)
            if srt[0] == 0 and all(v == 26 for v in srt[1:]):
                has_moon = True
            cur['scores'] = rs
            cur['totals'] = list(map(int, menv.match_scores))
            deals.append(cur)
            cur = {'pass_dir': None, 'plays': [], 'winners': []}
            trick = []
        if match_done:
            break
    return {'seed': seed, 'deals': deals,
            'final': list(map(int, menv.match_scores)),
            'placements': list(menv.placements())}, has_moon


def main():
    net = net_from_checkpoint(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'hearts_web_model.pth'))
    net.eval()
    cands = []
    for seed in CAND_SEEDS:
        m, moon = play_match(net, seed)
        n_deals = len(m['deals'])
        print(f'seed {seed}: {n_deals} deals, moon={moon}, '
              f'final {m["final"]}')
        cands.append((0 if moon else 1, n_deals, seed, m))
    cands.sort()
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, (_, nd, seed, m) in enumerate(cands[:3]):
        p = os.path.join(OUT_DIR, f'match_{i}.json')
        with open(p, 'w') as f:
            json.dump(m, f, separators=(',', ':'))
        print(f'wrote {p} (seed {seed}, {nd} deals, '
              f'{os.path.getsize(p) // 1024}KB)')


if __name__ == '__main__':
    main()
